import io

import lmdb
import numpy as np
import torch
import torch.utils.data as data

import utils


class LMDBDataset(data.Dataset):

    def __init__(self, root: str, split: str):
        super(LMDBDataset, self).__init__()
        assert split in ['train', 'valid', 'test']

        dset_dir = utils.get_path_to('data') / root
        assert dset_dir.is_dir(), \
            f'LMDBDataset @ {dset_dir} not found! Try generating it using the src/dataset/convert_to_lmdb.py script.'

        self.env = lmdb.open(
            str(dset_dir / f'{split}.lmdb'), subdir=False,
            max_readers=1,
            readonly=True, lock=False,
            readahead=False, meminit=False
        )
        with self.env.begin(write=False) as txn:
            self.length = txn.stat()['entries']

    def __getitem__(self, index: int):
        with self.env.begin(write=False, buffers=True) as txn:
            data_buffer = txn.get(str(index).encode())

        memfile = io.BytesIO()
        memfile.write(data_buffer)
        memfile.seek(0)
        npzfile = np.load(memfile)
        x, y = npzfile['x'], npzfile['y']
        return torch.from_numpy(x), torch.from_numpy(y)

    def __len__(self) -> int:
        return self.length
