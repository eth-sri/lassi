import argparse
import io
from typing import Optional

import lmdb
from loguru import logger
import numpy as np
import torch
from tqdm import tqdm

import args
from dataset.data_manager import DataManager
from models.gen_model_factory import GenModelFactory
import utils


def get_params() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a dataset (and save the Glow latents) into LMDB format",
        parents=[
            args.common(),
            args.dataset(),
            args.gen_model(),
            args.glow()
        ],
        conflict_handler='resolve'
    )
    return parser.parse_args()


def convert_and_save(params: argparse.Namespace, correlated_sens_attr: Optional[str] = None):
    device = utils.get_device(params.use_cuda)

    if correlated_sens_attr is None:
        data_dir = utils.get_path_to('data') / f'{params.gen_model_name}_latent_lmdb'
    else:
        data_dir = utils.get_path_to('data') / f'{params.gen_model_name}_latent_lmdb_correlated_{correlated_sens_attr}'
    dset_dir = utils.get_or_create_path(data_dir, empty_dir=True)

    gen_model = GenModelFactory.get_and_load(params, freeze=True).to(device)
    single_record_size_bytes = gen_model.latent_dimension() * 8 + 40 * 8

    data_manager = DataManager.get_manager(params)

    for split in ['train', 'valid', 'test']:
        logger.debug(f'Convert {split} subset.')
        BATCH_SIZE = 128
        data_loader = data_manager.get_dataloader(split, shuffle=False, batch_size=BATCH_SIZE)

        # Create LMDB:
        env = lmdb.open(
            str(dset_dir / f'{split}.lmdb'),
            map_size=single_record_size_bytes * len(data_loader) * BATCH_SIZE * 2,
            subdir=False,
            readonly=False
        )
        with env.begin(write=True) as txn:
            num_records = 0
            for x, y in tqdm(data_loader, ncols=150):
                assert gen_model is not None
                x = x.to(device)
                x = gen_model.latents(x)
                x = gen_model.wrap_latents(x).cpu()
                assert x.dim() == 2 and x.size(1) == gen_model.latent_dimension()

                if params.dataset == 'fairface':
                    y = torch.stack(y, dim=1)

                assert x.size(0) == y.size(0)

                for i in range(x.size(0)):
                    x_i = x[i].numpy()
                    y_i = y[i].numpy()

                    memfile = io.BytesIO()
                    np.savez(memfile, x=x_i, y=y_i)

                    txn.put(str(num_records).encode(), memfile.getvalue())
                    num_records += 1
        env.close()

        logger.debug(f'Converted {num_records} records')


if __name__ == '__main__':
    params = get_params()
    utils.init_logger()
    if params.dataset != '3dshapes':
        convert_and_save(params)
    else:
        assert params.dataset == '3dshapes'

        params.perturb = None
        convert_and_save(params)

        params.perturb = 'orientation'
        convert_and_save(params, params.perturb)
