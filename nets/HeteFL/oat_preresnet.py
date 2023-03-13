"""Ref to HeteroFL pre-activated ResNet18"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.batchnorm import _BatchNorm
from torch.nn.modules.instancenorm import _InstanceNorm
# from ..bn_ops import get_bn_layer
# from ..slimmable_models import BaseModule, SlimmableMixin
from nets.FiLM import FiLM2D
from nets.profile_func import profile_model
from ..oat_models import OATBaseModel
from ..slimmable_ops import SlimmableConv2d, SlimmableBatchNorm2d, SlimmableLinear


hidden_size = [64, 128, 256, 512]


class Block(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride, norm_layer, conv_layer, film_layer):
        super(Block, self).__init__()
        # self.norm_layer = norm_layer
        self.bn1 = norm_layer(in_planes)
        self.film1 = film_layer(in_planes)
        self.conv1 = conv_layer(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = norm_layer(planes)
        self.film2 = film_layer(planes)
        self.conv2 = conv_layer(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)

        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = conv_layer(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False)

    def forward(self, x):
        out = F.relu(self.film1(self.bn1(x)))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.film2(self.bn2(out))))
        out += shortcut
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride, norm_layer, conv_layer, film_layer):
        super(Bottleneck, self).__init__()
        # self.norm_layer = norm_layer
        self.bn1 = norm_layer(in_planes)
        self.film1 = film_layer(in_planes)
        self.conv1 = conv_layer(in_planes, planes, kernel_size=1, bias=False)
        self.bn2 = norm_layer(planes)
        self.film2 = film_layer(planes)
        self.conv2 = conv_layer(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn3 = norm_layer(planes)
        self.film3 = film_layer(planes)
        self.conv3 = conv_layer(planes, self.expansion * planes, kernel_size=1, bias=False)

        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = conv_layer(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False)

    def forward(self, x):
        out = F.relu(self.film1(self.bn1(x)))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.film2(self.bn2(out))))
        out = self.conv3(F.relu(self.film3(self.bn3(out))))
        out += shortcut
        return out


class ResNet(OATBaseModel):
    input_shape = [None, 3, 32, 32]

    def __init__(self, hidden_size, block, num_blocks, num_classes=10, bn_type='dbn',
                 lmbd_enc_dim=64):
        super(ResNet, self).__init__(lmbd_enc_dim)

        # if bn_type.startswith('d'):
        #     print("WARNING: When using dual BN, you should not do slimming.")
        # if track_running_stats:
        #     print("WARNING: We cannot track running_stats when slimmable BN is used.")
        # assert slimmable_layers == 'all', "Ignored kwargs"
        # assert generator_dim == 0, "Ignored kwargs"
        self.bn_type = bn_type
        # norm_layer = lambda n_ch: get_bn_layer(bn_type)['2d'](n_ch, track_running_stats=track_running_stats)
        if bn_type == 'bn':
            norm_layer = lambda n_ch: SlimmableBatchNorm2d(n_ch, track_running_stats=True)
        elif bn_type == 'dbn':
            from ..dual_bn import DualNormLayer
            norm_layer = lambda n_ch: DualNormLayer(n_ch, track_running_stats=True, affine=True, bn_class=SlimmableBatchNorm2d,
                 share_affine=False)
        else:
            raise RuntimeError(f"Not support bn_type={bn_type}")
        if lmbd_enc_dim > 0:
            film_layer = lambda n_ch: FiLM2D(n_ch, lmbd_enc_dim)
        else:
            film_layer = lambda n_ch: nn.Identity()
        conv_layer = SlimmableConv2d

        self.in_planes = hidden_size[0]
        self.conv1 = SlimmableConv2d(3, hidden_size[0], kernel_size=3, stride=1, padding=1, bias=False, non_slimmable_in=True)
        self.layer1 = self._make_layer(block, hidden_size[0], num_blocks[0], stride=1,
                                       norm_layer=norm_layer, conv_layer=conv_layer,
                                       film_layer=film_layer)
        self.layer2 = self._make_layer(block, hidden_size[1], num_blocks[1], stride=2,
                                       norm_layer=norm_layer, conv_layer=conv_layer,
                                       film_layer=film_layer)
        self.layer3 = self._make_layer(block, hidden_size[2], num_blocks[2], stride=2,
                                       norm_layer=norm_layer, conv_layer=conv_layer,
                                       film_layer=film_layer)
        self.layer4 = self._make_layer(block, hidden_size[3], num_blocks[3], stride=2,
                                       norm_layer=norm_layer, conv_layer=conv_layer,
                                       film_layer=film_layer)
        self.bn4 = norm_layer(hidden_size[3] * block.expansion)
        self.film4 = film_layer(hidden_size[3] * block.expansion)
        self.linear = SlimmableLinear(hidden_size[3] * block.expansion, num_classes, non_slimmable_out=True)

    def _make_layer(self, block, planes, num_blocks, stride, norm_layer, conv_layer,
                    film_layer):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, norm_layer, conv_layer, film_layer))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, return_pre_clf_fea=False):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.relu(self.film4(self.bn4(out)))
        out = F.adaptive_avg_pool2d(out, 1)
        out = out.view(out.size(0), -1)
        logits = self.linear(out)
        if return_pre_clf_fea:
            return logits, out
        else:
            return logits

    def print_footprint(self):
        input_shape = self.input_shape
        input_shape[0] = 2
        x = torch.rand(input_shape)
        batch = x.shape[0]
        print(f"input: {np.prod(x.shape[1:])} <= {x.shape[1:]}")
        x = self.conv1(x)
        print(f"conv1: {np.prod(x.shape[1:])} <= {x.shape[1:]}")
        for i_layer, layer in enumerate([self.layer1, self.layer2, self.layer3, self.layer4]):
            x = layer(x)
            print(f"layer {i_layer}: {np.prod(x.shape[1:]):5d} <= {x.shape[1:]}")

def init_param(m):
    if isinstance(m, (_BatchNorm, _InstanceNorm)):
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
        m.bias.data.zero_()
    return m


# Instantiations
def resnet18(**kwargs):
    model = ResNet(hidden_size, Block, [2, 2, 2, 2], **kwargs)
    model.apply(init_param)
    return model


def resnet34(**kwargs):
    model = ResNet(hidden_size, Block, [3, 4, 6, 3], **kwargs)
    model.apply(init_param)
    return model


def resnet50(**kwargs):
    model = ResNet(hidden_size, Bottleneck, [3, 4, 6, 3], **kwargs)
    model.apply(init_param)
    return model


def resnet101(**kwargs):
    model = ResNet(hidden_size, Bottleneck, [3, 4, 23, 3], **kwargs)
    model.apply(init_param)
    return model


def resnet152(**kwargs):
    model = ResNet(hidden_size, Bottleneck, [3, 8, 36, 3], **kwargs)
    model.apply(init_param)
    return model


def main():
    from nets.profile_func import profile_slimmable_models
    from nets.slimmable_models import Ensemble, EnsembleSubnet, EnsembleGroupSubnet

    print(f"profile model GFLOPs (forward complexity) and size (#param)")

    model = resnet18(bn_type='dbn')  # , lmbd_enc_dim=0)
    model.eval()  # this will affect bn etc

    print(f"model {model.__class__.__name__} on {'training' if model.training else 'eval'} mode")
    input_shape = model.input_shape
    model.set_bn_mode(0.4)

    film_params = 0
    total_params = 0
    # for n, p in model.named_parameters():
    for n, p in model.state_dict().items():
        if 'film' in n:
            # print(n)
            film_params += p.numel()
        total_params += p.numel()
    print(f"film_params: {film_params/1e6:g} / {total_params/1e6:g}MB, {film_params/total_params*100:.1f}%")

    flops, state_params = profile_model(model)
    print(f'GFLOPS: {flops / 1e9:.4f}, '
          f'model state size: {state_params / 1e6:.2f}MB')
    print(f"\n==footprint==")
    # model.switch_slim_mode(1.)
    model.print_footprint()
    print(f"\n==footprint==")
    # model.switch_slim_mode(0.125)
    model.print_footprint()


if __name__ == '__main__':
    main()

