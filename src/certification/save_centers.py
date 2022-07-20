import argparse
import time

from loguru import logger
import numpy as np
import torch
from tqdm import tqdm

import args
from certification.center_smoothing import CenterSmoothing
from certification.certification_utils import get_x_and_y
from dataset.data_manager import DataManager
from lassi_encoder.train_encoder import FairEncoderExperiment
from models.gen_model_factory import GenModelFactory
import utils

CERTIFIED_CENTERS_CACHE_DIR_NAME_TEMPLATE = 'centers_{}_enc_sigma_{}_alpha_{}_n_{}_n0_{}'


def get_params() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Certification (smoothing) evaluation pipeline",
        parents=[
            args.common(),
            args.dataset(),
            args.gen_model(),
            args.glow(),
            args.encoder(),
            args.fair_encoder(),
            args.certification()
        ],
        conflict_handler='resolve'
    )
    return parser.parse_args()


class CertifiedProducer:
    """
    Certified data producer: run center smoothing for the encoder and cache the certified centers and radii (save them
    on the disk).
    """

    def __init__(self, params: argparse.Namespace, data_manager: DataManager):
        self.device = utils.get_device(params.use_cuda)

        # Files and dirs:
        self.encoder_dir = utils.get_path_to('saved_models') / params.fair_encoder_name
        assert self.encoder_dir.is_dir(), f'Invalid encoder directory:\n{self.encoder_dir}'
        logger.debug(f'Encoder dir: {self.encoder_dir}')
        self.cache_dir = utils.get_or_create_path(
            self.encoder_dir / CERTIFIED_CENTERS_CACHE_DIR_NAME_TEMPLATE.format(
                utils.spec_id(params), params.enc_sigma, params.enc_alpha, params.enc_n, params.enc_n0
            )
        )
        logger.debug(f'Centers cache dir: {self.cache_dir}')

        # Data:
        self.data_manager = data_manager
        self.input_representation = params.input_representation
        # Data iteration:
        self.sampling_batch_size = params.sampling_batch_size
        self.skip = params.skip

        # Load models (required by the data producer):
        gen_model = GenModelFactory.get_and_load(params, freeze=True).to(self.device)
        lassi_encoder = FairEncoderExperiment.load_models(params, load_aux_classifier=False).to(self.device)

        # Smoothing:
        self.center_smoothing = CenterSmoothing(
            params, gen_model, lassi_encoder, params.enc_sigma,
            n_pred=params.enc_n0, n_cert=params.enc_n,
            alpha_1=params.enc_alpha / 2.0, alpha_2=params.enc_alpha / 2.0
        )
        self.perturb_epsilon = params.perturb_epsilon

    def compute_and_save_centers(self, subset: str):
        assert subset in ['valid', 'test']
        logger.debug(f'Compute centers on {subset} set.')
        data_loader = self.data_manager.get_dataloader(subset, shuffle=False)

        subset_dir = utils.get_or_create_path(self.cache_dir / f'{subset}')

        sample_index = 0
        radii = []
        smoothing_errors = []
        pbar = tqdm(total=len(data_loader), leave=False, ncols=150)
        for x, _ in data_loader:
            x = get_x_and_y(x, _, self.device, self.data_manager, skip=self.skip, get_y=False)

            for i in range(x.size(0)):
                if (subset_dir / f'{sample_index}.npz').is_file():
                    sample_index += self.skip
                    continue

                # Center smoothing:
                start_time = time.time()
                z_center, r1, smoothing_error = \
                    self.center_smoothing.certify(x[i], self.perturb_epsilon, self.sampling_batch_size)
                compute_time = time.time() - start_time

                if z_center is None:
                    z_center = np.array([])
                    assert r1 == CenterSmoothing.ABSTAIN
                else:
                    z_center = z_center.detach().cpu().numpy()
                    radii.append(r1)
                    smoothing_errors.append(smoothing_error)

                np.savez(
                    subset_dir / f'{sample_index}.npz',
                    z_center=z_center,
                    r1=np.array([r1]),
                    smoothing_error=smoothing_error,
                    t=np.array([compute_time])
                )
                sample_index += self.skip

            pbar.set_postfix(r_avg=f'{np.mean(radii):.4f}')
            pbar.update()
        pbar.close()
        logger.debug(f'Average radius: {np.mean(radii)}')
        logger.debug(f'Average smoothing error: {np.mean(smoothing_errors)}')


def main(params: argparse.Namespace):
    assert params.gen_model_name is not None
    assert params.fair_encoder_name is not None and params.perturb is not None and params.perturb_epsilon is not None
    assert params.batch_size % params.skip == 0

    utils.set_random_seed(params.seed)

    data_manager = DataManager.get_manager(params)
    certified_producer = CertifiedProducer(params, data_manager)

    with torch.no_grad():
        certified_producer.compute_and_save_centers(params.split)


if __name__ == '__main__':
    params = get_params()
    utils.init_logger()
    main(params)
