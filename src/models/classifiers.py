import argparse
from typing import List

import torch.nn as nn

from models.common import MLP
from models.lassi_encoder import LASSIEncoderFactory


class LinearClassifier(nn.Module):

    def __init__(self, input_dim: int, num_classes: int = 2):
        super(LinearClassifier, self).__init__()
        assert num_classes >= 2
        self.proj = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        """Output logits."""
        return self.proj(x)


class MLPClassifier(nn.Module):

    def __init__(self, input_dim: int, hidden_layers: List[int] = None, num_classes: int = 2):
        super(MLPClassifier, self).__init__()
        assert num_classes >= 2
        if hidden_layers is None:
            hidden_layers = []
        self.mlp = MLP([input_dim] + hidden_layers + [num_classes])

    def forward(self, x):
        return self.mlp(x)


def lassi_classifier_factory(params: argparse.Namespace, input_dim: int, num_classes: int) -> nn.Module:
    if not params.cls_layers:
        return LinearClassifier(input_dim, num_classes=num_classes)
    else:
        return MLPClassifier(input_dim, params.cls_layers, num_classes=num_classes)


class BaseClassifier(nn.Module):

    def __init__(self, latent_dim: int, num_classes: int = 2):
        super(BaseClassifier, self).__init__()
        self.out = LinearClassifier(latent_dim, num_classes=num_classes)

    def forward(self, x):
        r = self.latent_representation(x)
        return self.out(r)

    def latent_representation(self, x):
        raise NotImplementedError('`latent_representation` not implemented')


class OriginalImagesClassifier(BaseClassifier):

    def __init__(self, params: argparse.Namespace, latent_dim: int, num_classes: int = 2, nc: int = 3):
        super(OriginalImagesClassifier, self).__init__(latent_dim, num_classes=num_classes)
        self.lassi_encoder = LASSIEncoderFactory.get(params)

    def latent_representation(self, x):
        return self.lassi_encoder(x)
