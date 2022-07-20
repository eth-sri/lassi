import argparse
import json
import time
from pathlib import Path
import shutil
from typing import List

from loguru import logger
import numpy as np
import torch

import args
from certification.center_smoothing import CenterSmoothing
from certification.certification_utils import get_x_and_y, CertConsts, CertificationStatistics
from certification.classifier_smoothing import ClassifierSmoothing
from certification.save_centers import CERTIFIED_CENTERS_CACHE_DIR_NAME_TEMPLATE
from dataset.data_manager import DataManager
from lassi_classifier.train_classifier import FairClassifierTrainer
from lassi_encoder.train_encoder import FairEncoderExperiment
from models.gen_model_factory import GenModelFactory
import utils


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
            args.fair_classifier(),
            args.certification()
        ],
        conflict_handler='resolve'
    )
    return parser.parse_args()


class Certifier:
    """
    Certify the full pipeline combining the output of the center smoothing for the encoder and performing Cohen's
    smoothing for the classifier.
    """

    def __init__(self, params: argparse.Namespace, data_manager: DataManager):
        self.device = utils.get_device(params.use_cuda)

        # Files and dirs:
        self.encoder_dir = utils.get_path_to('saved_models') / params.fair_encoder_name
        assert self.encoder_dir.is_dir(), f'Invalid encoder directory:\n{self.encoder_dir}'
        self.centers_cache_dir = self.encoder_dir / CERTIFIED_CENTERS_CACHE_DIR_NAME_TEMPLATE.format(
            utils.spec_id(params), params.enc_sigma, params.enc_alpha, params.enc_n, params.enc_n0
        )
        assert self.centers_cache_dir.is_dir(), f'Invalid centers cache directory:\n{self.centers_cache_dir}'

        self.logs_dir = self.get_logs_dir(params)
        self.file_prefix = params.fair_classifier_name + f'_cls_alpha_{params.cls_alpha}'
        if params.save_artefacts:
            log_file = self.logs_dir / f'{self.file_prefix}.log'
            if log_file.is_file():
                log_file.unlink()
            self.log_file_id = utils.add_log_file(log_file)
        logger.debug(f'Logs dir: {self.logs_dir}')

        # Data:
        self.data_manager = data_manager
        self.input_representation = params.input_representation
        self.num_classes = self.data_manager.num_classes()
        assert params.split in ['valid', 'test']
        self.split = params.split
        # Data iteration:
        self.sampling_batch_size = params.sampling_batch_size
        self.skip = params.skip

        # Load models (all are frozen):
        self.gen_model = GenModelFactory.get_and_load(params, freeze=True).to(self.device)
        self.lassi_encoder = FairEncoderExperiment.load_models(params, load_aux_classifier=False).to(self.device)
        self.lassi_classifier = FairClassifierTrainer.load_model(params, freeze=True).to(self.device)

        # Smoothing:
        self.center_smoothing = CenterSmoothing(
            params, self.gen_model, self.lassi_encoder, params.enc_sigma,
            n_pred=params.enc_n0, n_cert=params.enc_n,
            alpha_1=params.enc_alpha / 2.0, alpha_2=params.enc_alpha / 2.0
        )
        self.perturb_epsilon = params.perturb_epsilon

        self.classifier_smoothing = ClassifierSmoothing(self.lassi_classifier, self.num_classes, params.cls_sigma)
        self.cls_alpha = params.cls_alpha
        self.cls_n = params.cls_n
        self.cls_n0 = params.cls_n0

        # For exploration and debugging purposes:
        self.perform_endpoints_analysis = params.perform_endpoints_analysis
        self.attr_vectors: List[List[torch.Tensor]] = []

    @staticmethod
    def get_logs_dir(params: argparse.Namespace) -> Path:
        return utils.get_or_create_path(
            utils.get_path_to('logs') / params.fair_encoder_name / params.split / (
                CERTIFIED_CENTERS_CACHE_DIR_NAME_TEMPLATE.format(
                    utils.spec_id(params), params.enc_sigma, params.enc_alpha, params.enc_n, params.enc_n0
                ) + f'_batch_size_{params.batch_size}_skip_{params.skip}'
            ),
            empty_dir=False
        )

    def certify_sample(self, sample_index: int, single_x: torch.Tensor, ground_truth_label) -> dict:
        # single_x is with its batch dimension stripped.
        certification_data = {CertConsts.SAMPLE_INDEX: sample_index}

        # Center smoothing:
        npz_file_path = self.centers_cache_dir / self.split / f'{sample_index}.npz'
        assert npz_file_path.is_file(), f'Center file missing: {npz_file_path}'
        npz_file = np.load(npz_file_path)
        z_center = torch.from_numpy(npz_file['z_center']).to(device=self.device, dtype=single_x.dtype)
        r1 = npz_file['r1'].item()
        smoothing_error = npz_file['smoothing_error'].item()
        center_smoothing_time = npz_file['t'].item()

        certification_data.update({
            CertConsts.CSM_RADIUS: r1,
            CertConsts.CSM_ERROR: smoothing_error,
            CertConsts.CSM_TIME: center_smoothing_time,
            CertConsts.GROUND_TRUTH_LABEL: ground_truth_label
        })

        if r1 >= 0:
            # Cohen's classifier smoothing:
            start_time = time.time()
            p, r2 = self.classifier_smoothing.certify(
                z_center.unsqueeze(0), self.cls_n0, self.cls_n, self.cls_alpha, self.sampling_batch_size
            )
            cohen_smoothing_time = time.time() - start_time

            if r1 < r2:
                r1_le_r2 = True
            else:
                r1_le_r2 = False

            certification_data.update({
                CertConsts.SMOOTHED_CLS_PRED: p,
                CertConsts.CERTIFIED_CLS_RADIUS: r2,
                CertConsts.CERTIFIED_FAIRNESS: (p != ClassifierSmoothing.ABSTAIN and r1_le_r2),
                CertConsts.COHEN_SMOOTHING_TIME: cohen_smoothing_time,
            })

            # Certified fairness + robustness:
            certification_data[CertConsts.CERTIFIED_FAIRNESS_AND_ACCURACY] = (p == ground_truth_label and r1_le_r2)
        else:
            certification_data.update({
                CertConsts.SMOOTHED_CLS_PRED: ClassifierSmoothing.ABSTAIN
            })

        return certification_data

    def evaluate_data(self):
        logger.debug(f'Collect evaluation data on {self.split} set.')
        data_loader = self.data_manager.get_dataloader(self.split, shuffle=False)
        eval_results_file = self.logs_dir / f'{self.file_prefix}_eval_results'
        eval_results_fh = eval_results_file.open('w')

        certification_statistics = CertificationStatistics()
        labels_count = [0] * self.num_classes

        sample_index = 0
        for x, y in data_loader:
            x, y = get_x_and_y(x, y, self.device, self.data_manager, self.skip, get_y=True)
            for i in range(x.size(0)):
                single_x = x[i]
                ground_truth_label = y[i].item()
                sample_eval_result = self.certify_sample(sample_index, single_x, ground_truth_label)

                if self.perform_endpoints_analysis:
                    sample_eval_result = self.endpoints_analysis(single_x, sample_eval_result)

                sample_index += self.skip
                certification_statistics.add(sample_eval_result)
                labels_count[ground_truth_label] += 1
                eval_results_fh.write(json.dumps(sample_eval_result) + '\n')
                eval_results_fh.flush()

        logger.debug('--- CERTIFICATION STATISTICS REPORT ---')
        logger.debug(f'Majority class: {100.0 * max(labels_count) / sum(labels_count):.2f}')
        print(eval_results_file)
        print(f'Majority class: {100.0 * max(labels_count) / sum(labels_count):.2f}')
        certification_statistics.report()
        print()

        eval_results_fh.close()
        shutil.copyfile(
            eval_results_file,
            self.logs_dir / f'{self.file_prefix}_eval_results_{certification_statistics.total_count()}'
        )

    def endpoints_analysis(self, single_x: torch.Tensor, sample_eval_data: dict) -> dict:
        z_gen_model_latents_along_segment = self.get_z_gen_model_latents_along_segment(single_x, get_center=False)

        sample_predictions = []
        for z_gen_model_latents in z_gen_model_latents_along_segment:
            z_lassi_center, _, _ = self.center_smoothing.certify(
                self.gen_model.wrap_latents(z_gen_model_latents).squeeze(0),
                self.perturb_epsilon, self.sampling_batch_size
            )
            if z_lassi_center is None:
                p = ClassifierSmoothing.ABSTAIN
            else:
                p, _ = self.classifier_smoothing.certify(
                    z_lassi_center.unsqueeze(0), self.cls_n0, self.cls_n, self.cls_alpha,
                    self.sampling_batch_size
                )
            sample_predictions.append(p)

        is_certified_fair = sample_eval_data.get(CertConsts.CERTIFIED_FAIRNESS, False)
        center_point_pred = sample_eval_data.get(CertConsts.SMOOTHED_CLS_PRED, ClassifierSmoothing.ABSTAIN)
        all_preds_match = (
            center_point_pred != ClassifierSmoothing.ABSTAIN and
            all(center_point_pred == p for p in sample_predictions)
        )

        sample_eval_data[CertConsts.EMPIRICALLY_FAIR] = all_preds_match
        sample_eval_data[CertConsts.CERTIFIED_FAIR_EMPIRICALLY_UNFAIR] = (is_certified_fair and not all_preds_match)

        return sample_eval_data

    def get_z_gen_model_latents_along_segment(self, single_x: torch.Tensor, get_center: bool = True) \
        -> List[List[torch.Tensor]]:
        if self.input_representation == 'original':
            z_gen_model_latents_original: List[torch.Tensor] = self.gen_model.latents(single_x.unsqueeze(0))
        else:
            assert self.input_representation == 'latent'
            z_gen_model_latents_original: List[torch.Tensor] = \
                self.gen_model.unwrap_latents(single_x.unsqueeze(0))

        if not self.attr_vectors:
            attr_vectors = self.gen_model.get_attribute_vectors(
                z_gen_model_latents_original[0].device, z_gen_model_latents_original[0].dtype
            )
            self.attr_vectors = [[z_i.unsqueeze(0) for z_i in attr_vector] for attr_vector in attr_vectors]

        z_gen_model_latents = []
        for attr_vector in self.attr_vectors:
            for plus_minus_one in [-1, 1]:
                z_gen_model_latents.append([
                    z_i + plus_minus_one * self.perturb_epsilon * z_delta_i
                    for z_i, z_delta_i in zip(z_gen_model_latents_original, attr_vector)
                ])

        if get_center:
            z_gen_model_latents.append(z_gen_model_latents_original)
        return z_gen_model_latents


def main(params: argparse.Namespace):
    assert params.gen_model_name is not None
    assert params.fair_encoder_name is not None and params.perturb is not None and params.perturb_epsilon is not None
    assert params.fair_classifier_name is not None
    assert params.batch_size % params.skip == 0

    utils.set_random_seed(params.seed)

    data_manager = DataManager.get_manager(params)
    certifier = Certifier(params, data_manager)

    with torch.no_grad():
        certifier.evaluate_data()

    if params.save_artefacts:
        logger.remove(certifier.log_file_id)


if __name__ == '__main__':
    params = get_params()
    utils.init_logger()
    main(params)
