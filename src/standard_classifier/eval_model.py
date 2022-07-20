import argparse
import json
import math

from loguru import logger
import numpy as np
import torch
from tqdm import tqdm

import args
from certification.certification_utils import get_x_and_y
from dataset.data_manager import DataManager
from lassi_classifier.train_classifier import FairClassifierTrainer
from lassi_encoder.train_encoder import FairEncoderExperiment
from metrics import ClassifierMetrics
from models.gen_model_factory import GenModelFactory, GenModelWrapper
from standard_classifier.train_model import StandardClassifierTrainer
import utils


def get_params() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Dataset statistics and fairness evaluation for a standard model",
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
    parser.add_argument('--compute_empirical_fairness', type=args.bool_flag, default=False,
                        help="Compute the empirical fairness of the classifier")
    return parser.parse_args()


class StandardModelEvaluator:

    def __init__(self, params: argparse.Namespace, data_manager: DataManager):
        self.device = utils.get_device(params.use_cuda)
        if params.fair_encoder_name is not None:
            # Can also be used to evaluate LASSI encoder + classifier model empirically, without smoothing.
            self.encoder = FairEncoderExperiment.load_models(params, load_aux_classifier=False).to(self.device)
            self.classifier = FairClassifierTrainer.load_model(params, freeze=True).to(self.device)
        else:
            self.encoder = None
            self.classifier = StandardClassifierTrainer.load_models(params, freeze=True).to(self.device)

        self.data_manager = data_manager
        self.gen_model = GenModelFactory.get_and_load(params, freeze=True).to(self.device)
        self.gen_model_wrapper = GenModelWrapper(params, self.gen_model)
        self.input_representation = params.input_representation
        self.use_gen_model_reconstructions = params.use_gen_model_reconstructions
        self.perturb = params.perturb
        self.perturb_epsilon = params.perturb_epsilon
        self.skip = params.skip
        self.compute_empirical_fairness = params.compute_empirical_fairness

    def compute_logits(self, enc_input):
        if self.encoder is None:
            return self.classifier(enc_input)
        else:
            z = self.encoder(enc_input)
            return self.classifier(z)

    def acc_and_empirical_fairness(self, subset: str):
        assert subset in ['valid', 'test']
        perturb_list = args.argslist_str(self.perturb)
        assert 1 <= len(perturb_list) <= 3

        data_loader = self.data_manager.get_dataloader(subset, shuffle=False)
        metrics = ClassifierMetrics(subset)

        if len(perturb_list) == 1:
            grid_step = self.perturb_epsilon / 4
            variations = np.arange(-self.perturb_epsilon, self.perturb_epsilon + 1e-5, grid_step).tolist()
            variation_coeffs = [[v] for v in variations]
        elif len(perturb_list) == 2:
            variation_coeffs = [[0.0, 0.0]]
            for plus_minus_one in [-1.0, 1.0]:
                variation_coeffs.append([plus_minus_one * self.perturb_epsilon, 0.0])
                variation_coeffs.append([0.0, plus_minus_one * self.perturb_epsilon])
            for pm1 in [-1.0, 1.0]:
                for pm2 in [-1.0, 1.0]:
                    variation_coeffs.append(
                        [pm1 * self.perturb_epsilon / math.sqrt(2), pm2 * self.perturb_epsilon / math.sqrt(2)]
                    )
        else:
            assert len(perturb_list) == 3
            variation_coeffs = [[0.0, 0.0, 0.0]]
            for plus_minus_one in [-1.0, 1.0]:
                variation_coeffs.append([plus_minus_one * self.perturb_epsilon, 0.0, 0.0])
                variation_coeffs.append([0.0, plus_minus_one * self.perturb_epsilon, 0.0])
                variation_coeffs.append([0.0, 0.0, plus_minus_one * self.perturb_epsilon])
            for pm1 in [-1.0, 1.0]:
                for pm2 in [-1.0, 1.0]:
                    for pm3 in [-1.0, 1.0]:
                        variation_coeffs.append([
                            pm1 * self.perturb_epsilon / math.sqrt(3),
                            pm2 * self.perturb_epsilon / math.sqrt(3),
                            pm3 * self.perturb_epsilon / math.sqrt(3),
                        ])

        for vs in variation_coeffs:
            assert len(vs) == len(perturb_list)
            assert np.linalg.norm(vs) <= self.perturb_epsilon

        total_counter = 0
        fairness_counter = 0

        with torch.no_grad():
            for x, y in tqdm(data_loader, ncols=125):
                if subset == 'test':
                    x, y = get_x_and_y(x, y, self.device, self.data_manager, self.skip, get_y=True)
                else:
                    y = self.data_manager.target_transform(y)
                    x = x.to(self.device)
                    y = y.to(self.device)

                z_gen_model_latents, enc_input = self.gen_model_wrapper.get_latents_and_encoder_input(x)
                logits = self.compute_logits(enc_input)
                y_pred = logits.argmax(dim=1).detach()
                metrics.add(batch_loss=0.0, y_targets=y, y_pred=y_pred)

                if self.compute_empirical_fairness:
                    individual_fairness_mask = torch.ones_like(y_pred, dtype=torch.bool)
                    attr_vectors = self.gen_model.get_attribute_vectors(
                        z_gen_model_latents[0].device,
                        z_gen_model_latents[0].dtype
                    )
                    for vs in variation_coeffs:
                        z_gen_model_latents_cur = [z_i.clone() for z_i in z_gen_model_latents]
                        for v, attr_vector in zip(vs, attr_vectors):
                            z_gen_model_latents_cur = [
                                z_i + v * z_delta_i
                                for z_i, z_delta_i in zip(z_gen_model_latents_cur, attr_vector)
                            ]
                        z_gen_model_latents_wrapped = self.gen_model.wrap_latents(z_gen_model_latents_cur)
                        enc_input = self.gen_model_wrapper.compute_encoder_input(z_gen_model_latents_wrapped)
                        logits = self.compute_logits(enc_input)
                        y_pred_grid = logits.argmax(dim=1)
                        individual_fairness_mask &= (y_pred == y_pred_grid)

                    fairness_counter += individual_fairness_mask.sum().item()
                    total_counter += x.size(0)

        logger.debug(f'Split: {subset}')
        logger.debug(f'[n = {metrics.total_samples}] {metrics.get_accuracies_str()}')
        acc, balanced_acc = metrics.get_accuracies()
        out_msg = {
            'split': subset,
            'n': metrics.total_samples,
            'acc': acc,
            'balanced_acc': balanced_acc
        }
        if self.compute_empirical_fairness:
            assert metrics.total_samples == total_counter
            logger.debug(f'Empirical individual fairness = {100.0 * fairness_counter / total_counter:.2f}')
            out_msg['empirical_fair'] = 100.0 * fairness_counter / total_counter

        print(json.dumps(out_msg))


def main(params: argparse.Namespace):
    assert params.gen_model_name is not None
    assert params.fair_classifier_name is not None and params.perturb is not None and params.perturb_epsilon is not None
    assert params.batch_size % params.skip == 0
    assert len(params.classify_attributes) == 1

    logger.debug(f'Dataset = {params.dataset}')
    data_manager = DataManager.get_manager(params)
    model_evaluator = StandardModelEvaluator(params, data_manager)

    logger.debug('Grid statistics...')
    model_evaluator.acc_and_empirical_fairness('valid')
    model_evaluator.acc_and_empirical_fairness('test')
    print()


if __name__ == '__main__':
    params = get_params()
    utils.common_experiment_setup(params.seed)
    main(params)
