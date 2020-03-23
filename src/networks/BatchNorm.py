import torch
import torch.nn as nn
import torch.nn.functional as F
from .distributions import VariationalPosterior, Prior
import math



class _BatchNorm(nn.Module):

    def __init__(self, num_features, args, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True, use_bias=True):
        super(_BatchNorm, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        self.use_bias = use_bias
        self.sig1 = args.sig1
        self.sig2 = args.sig2
        self.pi = args.pi
        self.rho = args.rho
        self.device = args.device

        if self.affine:
            # Weight parameters
            # self.weight_mu = nn.Parameter(torch.zeros(self.num_features).normal_(0., 0.1))
            # self.weight_rho = nn.Parameter(self.rho  + torch.zeros(self.num_features).normal_(0., 0.1))
            self.weight_mu = nn.Parameter(torch.empty((self.num_features),
                                                      device=self.device, dtype=torch.float32).normal_(0., 0.1),
                                          requires_grad=True)
            self.weight_rho = nn.Parameter(self.rho + torch.empty((self.num_features),
                                                                  device=self.device, dtype=torch.float32).normal_(0.,
                                                                                                                   0.1),
                                           requires_grad=True)

            self.weight = VariationalPosterior(self.weight_mu, self.weight_rho, self.device)
            
            # Bias parameters [out_channel]
            # self.bias_mu = nn.Parameter(torch.zeros(self.num_features).normal_(0., 0.1))
            # self.bias_rho = nn.Parameter(self.rho  + torch.zeros(self.num_features).normal_(0., 0.1))
            self.bias_mu = nn.Parameter(torch.empty((self.num_features),
                                      device=self.device, dtype=torch.float32).normal_(0., 0.1),requires_grad=True)
            self.bias_rho = nn.Parameter(self.rho + nn.Parameter(torch.empty(self.num_features,
                                      device=self.device, dtype=torch.float32).normal_(0., 0.1),requires_grad=True))

            self.bias = VariationalPosterior(self.bias_mu, self.bias_rho, self.device)

        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

        if self.track_running_stats:
            self.register_buffer('running_mean', torch.zeros(self.num_features).normal_(0., 1.))
            self.register_buffer('running_var', torch.zeros(self.num_features).normal_(0., 1.))
        else:
            self.register_parameter('running_mean', None)
            self.register_parameter('running_var', None)

        # Prior distributions
        self.weight_prior = Prior(args)
        self.bias_prior = Prior(args)
        
        self.log_prior = 0
        self.log_variational_posterior = 0
        
        self.mask_flag = False



 
    def prune_module(self, mask):
        self.mask_flag = True 
        self.pruned_weight_mu=self.weight_mu.data.mul_(mask)
        # self.pruned_weight_rho=self.weight_rho.data.mul_(mask)


    def _check_input_dim(self, input):
        return NotImplemented


    def forward(self, input, sample=False, calculate_log_probs=False):
        self._check_input_dim(input)
        if self.mask_flag:
            self.weight = VariationalPosterior(self.pruned_weight_mu, self.weight_rho, self.device)
            # if self.use_bias:
            #     self.bias = VariationalPosterior(self.bias_mu, self.bias_rho)



        if self.training or sample:
            weight = self.weight.sample()
            bias = self.bias.sample()
        else:
            weight = self.weight.mu
            bias = self.bias.mu

        if self.training or calculate_log_probs:
            self.log_prior = self.weight_prior.log_prob(weight) + self.bias_prior.log_prob(bias)
            self.log_variational_posterior = self.weight.log_prob(weight) + self.bias.log_prob(bias)
        else:
            self.log_prior, self.log_variational_posterior = 0, 0
        
        return F.batch_norm(input, self.running_mean, self.running_var, weight, bias,
                    self.training or not self.track_running_stats, self.momentum, self.eps)
        

    def extra_repr(self):
        return '{num_features}, eps={eps}, momentum={momentum}, affine={affine}, ' \
               'track_running_stats={track_running_stats}'.format(**self.__dict__)


class BayesianBatchNorm2d(_BatchNorm):

    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'
                             .format(input.dim()))
