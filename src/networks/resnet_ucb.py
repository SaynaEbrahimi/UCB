import torch
import torch.nn as nn
import torch.nn.functional as F

from .BayesianConvs import BayesianConv2D
from .BatchNorm import BayesianBatchNorm2d
from .FC import BayesianLinear



def conv3x3(in_planes, out_planes, args, stride=1):
    return BayesianConv2D(in_planes, out_planes, 3, args=args, stride=stride, padding=1, use_bias=True)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, args, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, args, stride)
        self.bn1 = BayesianBatchNorm2d(planes, args)
        # self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, args)
        self.bn2 = BayesianBatchNorm2d(planes, args)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x, sample=False):
        residual = x

        out = self.conv1(x, sample)
        out = self.bn1(out, sample)
        out = F.relu(out)

        out = self.conv2(out, sample)
        out = self.bn2(out, sample)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = F.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, args, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = BayesianConv2D(inplanes, planes, 1, args, use_bias=True)
        self.bn1 = BayesianBatchNorm2d(planes, args)
        self.conv2 = BayesianConv2D(planes, planes, 3, args, stride=stride, padding=1, use_bias=True)
        self.bn2 = BayesianBatchNorm2d(planes, args)
        self.conv3 = BayesianConv2D(planes, planes * self.expansion, args, kernel_size=1, use_bias=True)
        self.bn3 = BayesianBatchNorm2d(planes * self.expansion, args)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x, sample=False):
        residual = x

        out = self.conv1(x, sample)
        out = self.bn1(out, sample)
        out = F.relu(out, inplace=True)

        out = self.conv2(out, sample)
        out = self.bn2(out, sample)
        out = F.relu(out, inplace=True)

        out = self.conv3(out, sample)
        out = self.bn3(out, sample)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = F.relu(out, inplace=True)

        return out


class BayesianResNet(nn.Module):

    def __init__(self, block, layers, args):
        self.inplanes = 32
        super(BayesianResNet, self).__init__()

        self.args = args
        self.sig1 = args.sig1
        self.sig2 = args.sig2
        self.pi = args.pi
        self.rho = args.rho

        ncha, size, _ = args.inputsize
        self.taskcla = args.taskcla

        self.num_ftrs = 256 * block.expansion

        self.conv1 = BayesianConv2D(ncha, 32, 7, args, stride=2, padding=3,
                                    use_bias=True)
        self.bn1 = BayesianBatchNorm2d(32, args)
        self.layer1 = self._make_layer(block, 32, layers[0])
        self.layer2 = self._make_layer(block, 64, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 128, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 256, layers[3], stride=2)

        # self.fc = None
        self.classifier = torch.nn.ModuleList()
        for t, n in self.taskcla:
            self.classifier.append(BayesianLinear(self.num_ftrs, n, args))


    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                BayesianConv2D(self.inplanes, planes * block.expansion, 1, self.args,
                               stride=stride, use_bias=True),
                BayesianBatchNorm2d(planes * block.expansion, self.args),
            )

        layers = []
        layers.append(block(self.inplanes, planes, self.args, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, self.args))

        return nn.Sequential(*layers)

    def add_task(self, num_classes):
        self.classifier = BayesianLinear(self.num_ftrs, num_classes, self.args)

    def prune(self, mask_modules):
        for module, mask in mask_modules.items():
            module.prune_module(mask)

    def forward(self, x, sample=False):
        x = self.conv1(x, sample)
        x = self.bn1(x, sample)
        x = F.relu(x, inplace=True)
        x = F.max_pool2d(x, 3, 2, 1)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = x.view(x.size(0), -1)

        y = []
        for t, i in self.taskcla:
            y.append(self.classifier[t](x, sample))
        return [F.log_softmax(yy, dim=1) for yy in y]




# def ucb(experiment):
#     if experiment == 'mnist2':
#         rho = -3
#         sigma1 = 0.
#         sigma2 = 6.
#         pi = 0.25
#         samples = 1
#     elif experiment == 'pmnist':
#         rho = -3
#         sigma1 = 0.
#         sigma2 = 6.
#         pi = 0.25
#         samples = 1
#     elif experiment == 'cifar':
#         rho = -3
#         sigma1 = 0.
#         sigma2 = 6.
#         pi = 0.75
#         samples = 1
#     else:
#         rho = -3
#         sigma1 = 0.
#         sigma2 = 6.
#         pi = 0.75
#         samples = 1
#
#     return rho, sigma1, sigma2, pi


def Net(args):
    return BayesianResNet(BasicBlock, [3, 4, 6, 3], args)