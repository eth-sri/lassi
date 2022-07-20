import argparse

from loguru import logger
import torch
import torch.nn as nn
import torchvision

from models.common import BasicCNNCelebAEncoder, MLP
from models.gen_model_factory import GenModelFactory


class MyBatchNorm1d(nn.BatchNorm1d):
    # References:
    # https://github.com/ptrblck/pytorch_misc/blob/master/batch_norm_manual.py
    # https://d2l.ai/chapter_convolutional-modern/batch-norm.html

    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True):
        super(MyBatchNorm1d, self).__init__(num_features, eps, momentum, affine, track_running_stats)

    def forward(self, input):
        assert input.dim() == 2
        assert not self.affine

        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        # calculate running estimates
        if self.training:
            # use biased var in train
            var = input.var(dim=0, unbiased=False)
            n = input.size(0)
            with torch.no_grad():
                # update running_var with unbiased var
                self.running_var = exponential_average_factor * var * n / (n - 1)\
                    + (1 - exponential_average_factor) * self.running_var
        else:
            var = self.running_var

        input = input / (torch.sqrt(var + self.eps))
        return input


class LASSIEncoder(nn.Module):

    def __init__(self, params: argparse.Namespace):
        super(LASSIEncoder, self).__init__()

        if params.encoder_normalize_output:
            self.output_layer = nn.BatchNorm1d(self.latent_dimension(), affine=False)
            # self.output_layer = MyBatchNorm1d(self.latent_dimension(), affine=False)
        else:
            self.output_layer = nn.Identity()
        self.encoder = None

    @staticmethod
    def latent_dimension() -> int:
        raise NotImplementedError("LASSIEncoder's latent dimension method not implemented")

    def forward(self, images_batch):
        assert self.encoder is not None
        z = self.encoder(images_batch)
        return self.output_layer(z)


class LASSIEncoderBasicCNN(LASSIEncoder):

    def __init__(self, params: argparse.Namespace):
        super(LASSIEncoderBasicCNN, self).__init__(params)
        self.encoder = BasicCNNCelebAEncoder(params)

    @staticmethod
    def latent_dimension() -> int:
        return 256


class LASSIEncoderResNet(LASSIEncoder):

    def __init__(self, params: argparse.Namespace):
        super(LASSIEncoderResNet, self).__init__(params)
        self.encoder = torchvision.models.resnet18(pretrained=params.resnet_encoder_pretrained)
        self.encoder.fc = nn.Identity()

    @staticmethod
    def latent_dimension() -> int:
        return 512


class LASSIEncoderLinear(LASSIEncoder):

    def __init__(self, params: argparse.Namespace):
        super(LASSIEncoderLinear, self).__init__(params)
        self.encoder = MLP(
            layers=[GenModelFactory.get(params).latent_dimension()] + params.encoder_hidden_layers + [512],
            use_bn=params.encoder_use_bn
        )

    @staticmethod
    def latent_dimension() -> int:
        return 512


class LASSIEncoderFactory:

    @staticmethod
    def get(params: argparse.Namespace) -> LASSIEncoder:
        if params.encoder_type == 'basic_cnn':
            logger.debug('Build BasicCNN encoder')
            return LASSIEncoderBasicCNN(params)
        elif params.encoder_type == 'resnet':
            logger.debug('Build ResNet18 encoder')
            return LASSIEncoderResNet(params)
        elif params.encoder_type == 'linear':
            logger.debug('Build Linear encoder')
            return LASSIEncoderLinear(params)
        else:
            raise ValueError(f'Encoder type {params.encoder_type} is not supported')

    @staticmethod
    def get_latent_dimension(params: argparse.Namespace) -> int:
        if params.encoder_type == 'basic_cnn':
            return LASSIEncoderBasicCNN.latent_dimension()
        elif params.encoder_type == 'resnet':
            return LASSIEncoderResNet.latent_dimension()
        elif params.encoder_type == 'linear':
            return LASSIEncoderLinear.latent_dimension()
        else:
            raise ValueError(f'Encoder type {params.encoder_type} is not supported')
