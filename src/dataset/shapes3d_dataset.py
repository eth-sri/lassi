from typing import Optional

import h5py
from loguru import logger
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

import utils


SPLIT_FILE_NAME_TEMPLATE = 'custom_{}_split.npy'


class Shapes3dDataset(Dataset):

    FACTORS = ['floor_hue', 'wall_hue', 'object_hue', 'scale', 'shape', 'orientation']
    VALUES_PER_FACTOR = {'floor_hue': 10, 'wall_hue': 10, 'object_hue': 10, 'scale': 8, 'shape': 4, 'orientation': 15}

    # The target label is fixed to '(object_)hue'!
    TARGET = 'object_hue'

    IMAGES = None
    LABELS = None

    def __init__(self, root: str, split: str, transform, sens_attribute: Optional[str] = None):
        super(Shapes3dDataset, self).__init__()
        assert split in ['train', 'valid', 'test']

        self.dset_dir = utils.get_path_to('data') / root
        assert self.dset_dir.is_dir(), f'Shapes3dDataset @ {self.dset_dir} not found!'
        self.transform = transform

        if Shapes3dDataset.IMAGES is None:
            assert Shapes3dDataset.LABELS is None
            images_file = self.dset_dir / 'images.npy'
            labels_file = self.dset_dir / 'labels.npy'

            if not (images_file.exists() and labels_file.exists()):
                dataset = h5py.File(self.dset_dir / '3dshapes.h5', 'r')
                images = np.asarray(dataset.get('images'))
                labels = np.asarray(dataset.get('labels'))
                np.save(str(images_file), images)
                np.save(str(labels_file), labels)
            assert images_file.exists() and labels_file.exists()

            Shapes3dDataset.IMAGES = np.load(images_file)
            labels = np.load(labels_file)
            labels = pd.DataFrame(data=labels, columns=Shapes3dDataset.FACTORS)
            labels = labels[labels.columns].astype('category')
            for column in labels.columns:
                labels[column] = labels[column].cat.codes
            Shapes3dDataset.LABELS = labels.to_numpy().astype(np.int64)
        assert Shapes3dDataset.IMAGES is not None
        assert Shapes3dDataset.LABELS is not None

        assert (self.dset_dir / SPLIT_FILE_NAME_TEMPLATE.format(split)).is_file(), "Custom data split is missing!"
        self.indices = np.load(self.dset_dir / SPLIT_FILE_NAME_TEMPLATE.format(split))

        # Make train set correlated, if the sensitive attribute is not None and split is train.
        if sens_attribute is not None and split == 'train':
            label_0_sens_attr_0 = []
            label_1_sens_attr_0 = []
            label_0_sens_attr_1 = []
            label_1_sens_attr_1 = []

            num_labels = Shapes3dDataset.VALUES_PER_FACTOR[Shapes3dDataset.TARGET]
            all_labels = list(range(num_labels))
            label_0s = all_labels[:(num_labels >> 1)]
            label_1s = all_labels[(num_labels >> 1):]

            num_sens_attributes = Shapes3dDataset.VALUES_PER_FACTOR[sens_attribute]
            all_sens_attr_values = list(range(num_sens_attributes))
            sens_attr_0s = all_sens_attr_values[:(num_sens_attributes >> 1)]
            sens_attr_1s = all_sens_attr_values[(num_sens_attributes >> 1) + (num_sens_attributes % 2):]

            logger.debug('Target labels:')
            logger.debug(f'label 0s: {label_0s}')
            logger.debug(f'label 1s: {label_1s}')
            logger.debug(f'Sensitive attribute: {sens_attribute}')
            logger.debug(f'Sens attr 0s: {sens_attr_0s}')
            logger.debug(f'Sens attr 1s: {sens_attr_1s}')

            target_attr_idx = Shapes3dDataset.FACTORS.index(Shapes3dDataset.TARGET)
            assert target_attr_idx == 2
            sens_attr_idx = Shapes3dDataset.FACTORS.index(sens_attribute)
            for idx in self.indices:
                target_label_value = Shapes3dDataset.LABELS[idx][target_attr_idx]
                sens_attr_value = Shapes3dDataset.LABELS[idx][sens_attr_idx]
                if target_label_value in label_0s:
                    if sens_attr_value in sens_attr_0s:
                        label_0_sens_attr_0.append(idx)
                    elif sens_attr_value in sens_attr_1s:
                        label_0_sens_attr_1.append(idx)
                else:
                    assert target_label_value in label_1s
                    if sens_attr_value in sens_attr_0s:
                        label_1_sens_attr_0.append(idx)
                    elif sens_attr_value in sens_attr_1s:
                        label_1_sens_attr_1.append(idx)

            self.indices = []
            self.indices.extend(label_0_sens_attr_0)
            logger.debug(f'label = 0, sens_attr = 0: {len(label_0_sens_attr_0)}')
            self.indices.extend(label_1_sens_attr_1)
            logger.debug(f'label = 1, sens_attr = 1: {len(label_1_sens_attr_1)}')

    def __getitem__(self, index):
        idx = self.indices[index]
        image = Shapes3dDataset.IMAGES[idx]
        labels = Shapes3dDataset.LABELS[idx]
        image = self.transform(image)
        return image, labels

    def __len__(self):
        return len(self.indices)

    def get_similar(self, full_label, attribute):
        attr_idx = Shapes3dDataset.FACTORS.index(attribute)
        indices = np.where(
            np.equal(
                np.delete(Shapes3dDataset.LABELS, attr_idx, axis=1),
                np.delete(full_label.cpu().numpy(), attr_idx, axis=1)[0]
            ).all(axis=1)
        )
        images = Shapes3dDataset.IMAGES[indices]
        labels = Shapes3dDataset.LABELS[indices]
        images = torch.cat([
            self.transform(image).unsqueeze(0) for image in images
        ])
        return images, labels
