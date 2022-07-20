import csv
from pathlib import Path

import numpy as np
import PIL.Image
from sklearn.model_selection import train_test_split
import torch.utils.data as data

import utils

TRAIN_SIZE_PROPORTION = 0.8
TRAIN_VALID_SPLIT_RNG = np.random.RandomState(42)
CSV_FILE_NAME_TEMPLATE = 'fairface_label_{}.csv'


# Supported attributes: Age, Age_bin ([0-29], [30+] ranges), Age_3 ([0-19], [20-39], [40+]), Race.


class FairFaceDataset(data.Dataset):

    FAIR_FACE_AGE_RANGES = ['0-2', '3-9', '10-19', '20-29', '30-39', '40-49', '50-59', '60-69', 'more than 70']
    FAIR_FACE_RACES = ['Black', 'East_Asian', 'Indian', 'Latino_Hispanic', 'Middle_Eastern', 'Southeast_Asian', 'White']

    TRAIN_SPLIT_SAMPLES = None
    VALID_SPLIT_SAMPLES = None

    def __init__(self, root: str, split: str, transform):
        super(FairFaceDataset, self).__init__()
        assert split in ['train', 'valid', 'test']

        self.dset_dir = utils.get_path_to('data') / root
        assert self.dset_dir.is_dir(), f'FairFaceDataset @ {self.dset_dir} not found!'
        self.transform = transform

        if split == 'train':
            self.split_samples, _ = self.train_valid_split_samples()
        elif split == 'valid':
            _, self.split_samples = self.train_valid_split_samples()
        elif split == 'test':
            self.split_samples = self.load_csv(self.dset_dir / CSV_FILE_NAME_TEMPLATE.format('val'))

    def train_valid_split_samples(self):
        if FairFaceDataset.TRAIN_SPLIT_SAMPLES is None:
            assert FairFaceDataset.VALID_SPLIT_SAMPLES is None
            train_valid_samples = self.load_csv(self.dset_dir / CSV_FILE_NAME_TEMPLATE.format('train'))
            if (self.dset_dir / 'custom_train_split.txt').is_file() and \
                    (self.dset_dir / 'custom_valid_split.txt').is_file():
                train_ids = self.read_custom_split_ids('train')
                valid_ids = self.read_custom_split_ids('valid')
                FairFaceDataset.TRAIN_SPLIT_SAMPLES = [train_valid_samples[id - 1] for id in train_ids]
                FairFaceDataset.VALID_SPLIT_SAMPLES = [train_valid_samples[id - 1] for id in valid_ids]
            else:
                FairFaceDataset.TRAIN_SPLIT_SAMPLES, FairFaceDataset.VALID_SPLIT_SAMPLES = train_test_split(
                    train_valid_samples,
                    train_size=TRAIN_SIZE_PROPORTION, random_state=TRAIN_VALID_SPLIT_RNG, shuffle=True
                )
        assert FairFaceDataset.TRAIN_SPLIT_SAMPLES is not None and FairFaceDataset.VALID_SPLIT_SAMPLES is not None
        return FairFaceDataset.TRAIN_SPLIT_SAMPLES, FairFaceDataset.VALID_SPLIT_SAMPLES

    @staticmethod
    def load_csv(csv_file: Path):
        rows = []
        with open(csv_file, newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                row.pop('service_test')
                row['age'] = FairFaceDataset.FAIR_FACE_AGE_RANGES.index(row['age'])
                row['race'] = FairFaceDataset.FAIR_FACE_RACES.index(row['race'].replace(' ', '_'))
                rows.append(row)
        return rows

    def read_custom_split_ids(self, split_name: str):
        assert split_name in ['train', 'valid']
        with open(self.dset_dir / f'custom_{split_name}_split.txt', 'r') as f:
            lines = f.readlines()
            ids = [int(l.rstrip()) for l in lines]
        return ids

    def __getitem__(self, index: int):
        x = PIL.Image.open(self.dset_dir / self.split_samples[index]['file'])
        x = self.transform(x)
        return x, (self.split_samples[index]['age'], self.split_samples[index]['race'])

    def __len__(self) -> int:
        return len(self.split_samples)
