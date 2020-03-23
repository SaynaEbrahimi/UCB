import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair

from .distributions import VariationalPosterior, Prior



class _ConvNd(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation, transposed, output_padding, groups, use_bias, args):
        super(_ConvNd, self).__init__()
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.transposed = transposed
        self.output_padding = output_padding
        self.groups = groups
        self.use_bias = use_bias
        self.sig1 = args.sig1
        self.sig2 = args.sig2
        self.pi = args.pi
        self.rho = args.rho
        self.device = args.device


        if transposed:
            self.weight_mu = nn.Parameter(torch.Tensor(in_channels, out_channels//groups, *kernel_size).normal_(0., 0.1))
            # self.weight_mu = nn.Parameter(torch.normal(mean=0., std=0.1, size=(in_channels, out_channels//groups, *kernel_size)))
            self.weight_rho = nn.Parameter(self.rho + torch.zeros(in_channels, out_channels//groups,*kernel_size).normal_(0., 0.1))

        else:
            # self.weight_mu = nn.Parameter(torch.Tensor(out_channels, in_channels//groups, *kernel_size).normal_(0., 0.1))

            self.weight_mu = nn.Parameter(torch.empty((out_channels, in_channels//groups, *kernel_size),
                                     device=self.device, dtype=torch.float32).normal_(0., 0.1), requires_grad=True)
            self.weight_rho = nn.Parameter(self.rho + torch.empty((out_channels, in_channels//groups, *kernel_size),
                                        device=self.device, dtype=torch.float32).normal_(0.,0.1), requires_grad=True)

            # self.weight_mu = nn.Parameter(torch.normal(mean=0., std=0.1, size=(out_channels, in_channels//groups, *kernel_size)))
            # self.weight_rho = nn.Parameter(self.rho + torch.zeros(out_channels, in_channels//groups,*kernel_size).normal_(0., 0.1))
                
        self.weight = VariationalPosterior(self.weight_mu, self.weight_rho, self.device).to(self.device)

        
        # Bias parameters [out_channel]
        if self.use_bias:
            # self.bias_mu = nn.Parameter(torch.Tensor(self.out_channels).normal_(0., 0.1))
            # self.bias_mu = nn.Parameter(torch.zeros(self.out_channels).normal_(0., 0.1))
            # self.bias_rho = nn.Parameter(self.rho + torch.zeros(self.out_channels).normal_(0., 0.1))
            self.bias_mu = nn.Parameter(torch.empty((self.out_channels),
                                      device=self.device, dtype=torch.float32).normal_(0., 0.1),requires_grad=True)
            self.bias_rho = nn.Parameter(self.rho + nn.Parameter(torch.empty(self.out_channels,
                                      device=self.device, dtype=torch.float32).normal_(0., 0.1),requires_grad=True))

            self.bias = VariationalPosterior(self.bias_mu, self.bias_rho, self.device).to(self.device)
        else:
            self.register_parameter('bias', None)            
        
        # Prior distributions
        self.weight_prior = Prior(args).to(self.device)

        if self.use_bias:      
            self.bias_prior = Prior(args).to(self.device)

        self.log_prior = 0
        self.log_variational_posterior = 0
        
        self.mask_flag = False


class BayesianConv2D(_ConvNd):

    def __init__(self, in_channels, out_channels, kernel_size, args, stride=1, padding=0, dilation=1, groups=1, use_bias=True):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(BayesianConv2D, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation, False, _pair(0), groups, use_bias, args)



    def prune_module(self, mask):
        self.mask_flag = True 
        self.pruned_weight_mu=self.weight_mu.data.mul_(mask)
        # self.pruned_weight_rho=self.weight_rho.data.mul_(mask)
        # pruning_mask = torch.eq(mask, torch.zeros_like(mask))


    def forward(self, input, sample=False, calculate_log_probs=False):
        if self.mask_flag:
            self.weight = VariationalPosterior(self.pruned_weight_mu, self.weight_rho, self.device)
            # if self.use_bias:
            #     self.bias = VariationalPosterior(self.bias_mu, self.bias_rho)

        if self.training or sample:
            weight = self.weight.sample()
            bias = self.bias.sample() if self.use_bias else None
                
        else:
            weight = self.weight.mu
            bias = self.bias.mu if self.use_bias else None

        if self.training or calculate_log_probs:
            if self.use_bias:
                self.log_prior = self.weight_prior.log_prob(weight) + self.bias_prior.log_prob(bias)
                self.log_variational_posterior = self.weight.log_prob(weight) + self.bias.log_prob(bias)
            else:
                self.log_prior = self.weight_prior.log_prob(weight)
                self.log_variational_posterior = self.weight.log_prob(weight)
            
        else:
            self.log_prior, self.log_variational_posterior = 0, 0
        
        return F.conv2d(input, weight, bias, self.stride, self.padding, self.dilation, self.groups)
