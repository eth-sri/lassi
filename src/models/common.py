import argparse
from typing import List

import torch
import torch.nn as nn


def fc_block(in_features: int, out_features: int, use_bn: bool = True, bn_affine: bool = True, use_relu: bool = True) \
        -> nn.ModuleList:
    module_list = nn.ModuleList([nn.Linear(in_features, out_features)])
    if use_bn:
        module_list.append(nn.BatchNorm1d(out_features, affine=bn_affine))
    if use_relu:
        module_list.append(nn.ReLU(True))
    return module_list


def conv_block(in_channels: int, out_channels: int, kernel_size: int, stride: int, padding: int = 0,
               use_bn: bool = True) -> nn.ModuleList:
    module_list = nn.ModuleList([nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)])
    if use_bn:
        module_list.append(nn.BatchNorm2d(out_channels))
    module_list.append(nn.ReLU(True))
    return module_list


class View(nn.Module):
    def __init__(self, size):
        super(View, self).__init__()
        self.size = size

    def forward(self, tensor):
        return tensor.view(self.size)


class MLP(nn.Module):

    def __init__(self, layers: List[int], use_bn: bool = True):
        super(MLP, self).__init__()
        self.module_list = nn.ModuleList()
        for in_dim, out_dim in zip(layers[:-2], layers[1:-1]):
            self.module_list.extend(fc_block(in_dim, out_dim, use_bn=use_bn, use_relu=True))
        self.module_list.extend(fc_block(layers[-2], layers[-1], use_bn=False, use_relu=False))
        self.mlp = nn.Sequential(*self.module_list)

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        return self.mlp(x)


class BasicCNNCelebAEncoder(nn.Module):
    """
    General CelebA encoder, based on the architecture described in the BetaVAE paper.
    The output/normalization layer is supposed to be defined in the LASSIEncoder model.
    """

    def __init__(self, params: argparse.Namespace, nc: int = 3):
        super(BasicCNNCelebAEncoder, self).__init__()

        self.module_list = nn.ModuleList()
        # Use padding = 1?
        self.module_list.extend(conv_block(nc, 32, 4, 2, 1, use_bn=params.encoder_use_bn))
        self.module_list.extend(conv_block(32, 32, 4, 2, 1, use_bn=params.encoder_use_bn))
        self.module_list.extend(conv_block(32, 64, 4, 2, 1, use_bn=params.encoder_use_bn))
        self.module_list.extend(conv_block(64, 64, 4, 2, 1, use_bn=params.encoder_use_bn))
        self.module_list.append(View((-1, 64 * 4 * 4)))

        if params.encoder_normalize_output:
            self.module_list.extend(fc_block(64 * 4 * 4, 256, use_bn=False, use_relu=False))
        else:
            # When NOT normalizing the output:
            # - if Batch Normalization is used, it is applied with learnable affine parameters (bn_affine=True);
            # - otherwise (Batch Normalization will not be used), bn_affine=True is irrelevant.
            self.module_list.extend(
                fc_block(64 * 4 * 4, 256, use_bn=params.encoder_use_bn, bn_affine=True, use_relu=False)
            )

        self.encoder = nn.Sequential(*self.module_list)

    def forward(self, images_batch):
        return self.encoder(images_batch)


class LinearDecoder(nn.Module):

    def __init__(self, params: argparse.Namespace, input_dimension: int, output_dimension: int):
        super(LinearDecoder, self).__init__()
        self.mlp = MLP([input_dimension] + params.recon_decoder_layers + [output_dimension])

    def forward(self, x):
        return self.mlp(x)
