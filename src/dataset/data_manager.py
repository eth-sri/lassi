from abc import ABC, abstractmethod
import argparse
from functools import partial
from typing import Optional

import torch
from torch.utils.data import DataLoader
from torchvision.datasets import CelebA
from torchvision import transforms

from dataset.fairface_dataset import FairFaceDataset
from dataset.lmdb_dataset import LMDBDataset
from dataset.shapes3d_dataset import Shapes3dDataset
import utils

CELEBA_ATTR_NAMES = [
    '5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive', 'Bags_Under_Eyes', 'Bald',
    'Bangs', 'Big_Lips', 'Big_Nose', 'Black_Hair', 'Blond_Hair',
    'Blurry', 'Brown_Hair', 'Bushy_Eyebrows', 'Chubby', 'Double_Chin',
    'Eyeglasses', 'Goatee', 'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones',
    'Male', 'Mouth_Slightly_Open', 'Mustache', 'Narrow_Eyes', 'No_Beard',
    'Oval_Face', 'Pale_Skin', 'Pointy_Nose', 'Receding_Hairline', 'Rosy_Cheeks',
    'Sideburns', 'Smiling', 'Straight_Hair', 'Wavy_Hair', 'Wearing_Earrings',
    'Wearing_Hat', 'Wearing_Lipstick', 'Wearing_Necklace', 'Wearing_Necktie', 'Young'
]


def glow_preprocessing_transform(n_bits, x: torch.Tensor):
    n_bins = 2.0 ** n_bits
    x = x * 255
    if n_bits < 8:
        x = torch.floor(x / 2 ** (8 - n_bits))
    x = x / n_bins - 0.5
    return x


def original_image_transforms(image_size: int, n_bits):
    transformations = [
        transforms.ToTensor(),  # Maps to the [0, 1] range.
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        # Glow preprocessing transform:
        partial(glow_preprocessing_transform, n_bits)
    ]
    return transforms.Compose(transformations)


class DataManager(ABC):

    def __init__(self, params: argparse.Namespace):
        self.params = params
        self.dataset_cache = {}
        self.loaders_cache = {}

    @staticmethod
    def get_manager(params: argparse.Namespace) -> 'DataManager':
        if params.dataset == 'celeba':
            return CelebAOriginalDataManager(params)
        elif params.dataset in ['glow_celeba_64_latent_lmdb', 'glow_celeba_128_latent_lmdb']:
            return CelebALMDBDataManager(params)
        elif params.dataset == 'fairface':
            return FairFaceOriginalDataManager(params)
        elif params.dataset == 'glow_fairface_latent_lmdb':
            return FairFaceLMDBDataManager(params)
        elif params.dataset == '3dshapes':
            return Shapes3dOriginalDataManager(params)
        elif params.dataset in [
            'glow_3dshapes_latent_lmdb',
            'glow_3dshapes_latent_lmdb_correlated_orientation',
        ]:
            return Shapes3dLMDBDataManager(params)
        else:
            raise ValueError(f'Dataset {params.dataset} not supported')

    def get_dataset(self, split: str):
        assert split in ['train', 'valid', 'test']
        if split not in self.dataset_cache:
            self.dataset_cache[split] = self._get_dataset(split)
        return self.dataset_cache[split]

    @abstractmethod
    def _get_dataset(self, split: str):
        raise NotImplementedError('`get_dataset` not implemented')

    def get_dataloader(self, split: str, shuffle: bool = True, batch_size: Optional[int] = None) -> DataLoader:
        assert split in ['train', 'valid', 'test']
        shuffle &= (split == 'train')
        if batch_size is None:
            batch_size = self.params.batch_size
        k = (split, shuffle, batch_size)
        if k not in self.loaders_cache:
            self.loaders_cache[k] = DataLoader(
                self.get_dataset(split),
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=self.params.num_workers,
                pin_memory=True
            )
        return self.loaders_cache[k]

    @abstractmethod
    def num_classes(self) -> int:
        raise NotImplementedError('`num_classes` not implemented')

    @abstractmethod
    def target_transform(self, y):
        raise NotImplementedError('`target_transform` not implemented')


class CelebADataManager(DataManager):

    def __init__(self, params: argparse.Namespace):
        super(CelebADataManager, self).__init__(params)
        self.attr_indices = torch.tensor([CELEBA_ATTR_NAMES.index(attr) for attr in params.classify_attributes])

    def _get_dataset(self, split: str):
        raise NotImplementedError('`get_dataset` not implemented')

    def num_classes(self) -> int:
        return (1 << len(self.params.classify_attributes))

    def target_transform(self, y):
        y = torch.index_select(y, 1, self.attr_indices).squeeze(dim=1)
        return y


class CelebAOriginalDataManager(CelebADataManager):

    def __init__(self, params: argparse.Namespace):
        super(CelebAOriginalDataManager, self).__init__(params)

    def _get_dataset(self, split: str):
        return CelebA(
            root=str(utils.get_path_to('data')),
            split=split,
            target_type='attr',
            transform=original_image_transforms(self.params.image_size, self.params.n_bits),
            download=True
        )


class CelebALMDBDataManager(CelebADataManager):

    def __init__(self, params: argparse.Namespace):
        super(CelebALMDBDataManager, self).__init__(params)

    def _get_dataset(self, split: str):
        return LMDBDataset(self.params.dataset, split)


class FairFaceDataManager(DataManager):

    def __init__(self, params: argparse.Namespace):
        super(FairFaceDataManager, self).__init__(params)
        self.classify_attributes = params.classify_attributes

    def _get_dataset(self, split: str):
        raise NotImplementedError('`get_dataset` not implemented')

    def num_classes(self) -> int:
        if 'Age' in self.params.classify_attributes:
            assert 'Age_bin' not in self.params.classify_attributes
            assert 'Age_3' not in self.params.classify_attributes
        if 'Age_bin' in self.params.classify_attributes:
            assert 'Age' not in self.params.classify_attributes
            assert 'Age_3' not in self.params.classify_attributes
        if 'Age_3' in self.params.classify_attributes:
            assert 'Age_bin' not in self.params.classify_attributes
            assert 'Age' not in self.params.classify_attributes

        prod = 1
        if 'Age' in self.params.classify_attributes:
            prod *= len(FairFaceDataset.FAIR_FACE_AGE_RANGES)
        if 'Age_bin' in self.params.classify_attributes:
            prod *= 2
        if 'Age_3' in self.params.classify_attributes:
            prod *= 3
        if 'Race' in self.params.classify_attributes:
            prod *= len(FairFaceDataset.FAIR_FACE_RACES)

        return prod

    @staticmethod
    def _get_attribute(y, attr: str):
        raise NotImplementedError('`_get_attribute` not implemented')

    def target_transform(self, y):
        if len(self.classify_attributes) == 1:
            return self._get_attribute(y, self.classify_attributes[0])
        elif len(self.classify_attributes) == 2:
            col0 = self._get_attribute(y, self.classify_attributes[0])
            col1 = self._get_attribute(y, self.classify_attributes[1])
            return torch.stack([col0, col1], dim=1)
        else:
            raise ValueError(f'Invalid classification attributes list: {self.classify_attributes}')


class FairFaceOriginalDataManager(FairFaceDataManager):

    def __init__(self, params: argparse.Namespace):
        super(FairFaceOriginalDataManager, self).__init__(params)

    def _get_dataset(self, split: str):
        return FairFaceDataset(
            self.params.dataset, split,
            original_image_transforms(self.params.image_size, self.params.n_bits)
        )

    @staticmethod
    def _get_attribute(y, attr: str):
        if attr == 'Age':
            return y[0]
        elif attr == 'Age_bin':
            y_ret = y[0]
            y_ret = (y_ret > 3).long()
            return y_ret
        elif attr == 'Age_3':
            y_ret = y[0]
            y_ret = torch.where(y_ret <= 2, 0, y_ret)
            y_ret = torch.where(y_ret >= 5, 2, y_ret)
            y_ret = torch.where(y_ret >= 3, 1, y_ret)
            return y_ret
        elif attr == 'Race':
            return y[1]
        else:
            raise ValueError(f'Invalid attribute name: {attr}')


class FairFaceLMDBDataManager(FairFaceDataManager):

    def __init__(self, params: argparse.Namespace):
        super(FairFaceLMDBDataManager, self).__init__(params)

    def _get_dataset(self, split: str):
        return LMDBDataset(self.params.dataset, split)

    @staticmethod
    def _get_attribute(y, attr: str):
        if attr == 'Age':
            return y[:, 0]
        elif attr == 'Age_bin':
            y_ret = y[:, 0]
            y_ret = (y_ret > 3).long()
            return y_ret
        elif attr == 'Age_3':
            y_ret = y[:, 0]
            y_ret = torch.where(y_ret <= 2, 0, y_ret)
            y_ret = torch.where(y_ret >= 5, 2, y_ret)
            y_ret = torch.where(y_ret >= 3, 1, y_ret)
            return y_ret
        elif attr == 'Race':
            return y[:, 1]
        else:
            raise ValueError(f'Invalid attribute name: {attr}')


class Shapes3dDataManager(DataManager):

    def __init__(self, params: argparse.Namespace):
        super(Shapes3dDataManager, self).__init__(params)
        self.attr_indices = torch.tensor([Shapes3dDataset.FACTORS.index(attr) for attr in params.classify_attributes])

    def _get_dataset(self, split: str):
        raise NotImplementedError('`get_dataset` not implemented')

    def num_classes(self) -> int:
        assert len(set(self.params.classify_attributes)) == len(self.params.classify_attributes)
        prod = 1
        for attr in self.params.classify_attributes:
            prod *= Shapes3dDataset.VALUES_PER_FACTOR[attr]
        return prod

    def target_transform(self, y):
        y = torch.index_select(y, 1, self.attr_indices).squeeze(dim=1)
        return y


class Shapes3dOriginalDataManager(Shapes3dDataManager):

    def __init__(self, params: argparse.Namespace):
        super(Shapes3dOriginalDataManager, self).__init__(params)
        if params.perturb is None or params.perturb in ['orientation']:
            self.sens_attribute = params.perturb
        else:
            raise ValueError(f'The sensitive attribute {params.perturb} is not supported by the 3dshapes dataset')

    def _get_dataset(self, split: str):
        return Shapes3dDataset(
            self.params.dataset, split,
            original_image_transforms(self.params.image_size, self.params.n_bits),
            self.sens_attribute
        )


class Shapes3dLMDBDataManager(Shapes3dDataManager):

    def __init__(self, params: argparse.Namespace):
        super(Shapes3dLMDBDataManager, self).__init__(params)

    def _get_dataset(self, split: str):
        return LMDBDataset(self.params.dataset, split)
