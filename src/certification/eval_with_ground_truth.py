import argparse
from collections import Counter

from loguru import logger
from matplotlib import pyplot as plt
from torchvision.utils import make_grid
import torch

import args
from certification.center_smoothing import CenterSmoothing
from certification.certification_utils import get_x_and_y
from certification.classifier_smoothing import ClassifierSmoothing
from dataset.data_manager import DataManager
from dataset.shapes3d_dataset import Shapes3dDataset
from lassi_classifier.train_classifier import FairClassifierTrainer
from lassi_encoder.train_encoder import FairEncoderExperiment
from models.gen_model_factory import GenModelFactory
import utils


def get_params() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Attribute vector investigation and ground-truth fairness evaluation",
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
    parser.add_argument('--perturb_factor', type=str,
                        help="Perturb factor (if different from the attribute vector name)")
    parser.add_argument('--save_reconstruction_images', type=args.bool_flag, default=False,
                        help="Save the Glow reconstructions together with the ground-truths")
    return parser.parse_args()


def main(params: argparse.Namespace):
    assert len(params.classify_attributes) == 1
    assert params.gen_model_name is not None
    assert params.fair_encoder_name is not None
    assert params.fair_classifier_name is not None
    assert params.dataset == '3dshapes'
    assert params.perturb_factor is None
    params.perturb_factor = params.perturb
    assert params.certification_batch_size == params.skip

    factor_values = Shapes3dDataset.VALUES_PER_FACTOR[params.perturb_factor]

    device = utils.get_device(params.use_cuda)
    data_manager = DataManager.get_manager(params)

    gen_model = GenModelFactory.get_and_load(params, freeze=True).to(device)
    attr_vectors = gen_model.get_attribute_vectors(device, torch.float)
    attr_vector = attr_vectors[4]  # Visualize with orientation_0_14.npz
    num_classes = data_manager.num_classes()
    lassi_encoder = FairEncoderExperiment.load_models(params, load_aux_classifier=False, freeze=True).to(device)
    lassi_classifier = FairClassifierTrainer.load_model(params, freeze=True).to(device)
    classifier_smoothing = ClassifierSmoothing(lassi_classifier, num_classes, params.cls_sigma)
    center_smoothing = CenterSmoothing(
        params, gen_model, lassi_encoder, params.enc_sigma,
        n_pred=params.enc_n0, n_cert=params.enc_n,
        alpha_1=params.enc_alpha / 2.0, alpha_2=params.enc_alpha / 2.0
    )

    file_or_dir = utils.get_path_to('saved_models') / params.gen_model_name
    attr_vectors_dir = utils.get_or_create_path(
        utils.get_directory_of_path(file_or_dir) / params.attr_vectors_dir / params.split
    )

    data_loader = data_manager.get_dataloader(
        split=params.split, shuffle=False, batch_size=params.certification_batch_size
    )
    sample_idx = 0
    cnt_emp_unfair, cnt_total, cnt_any_certified_but_unfair, cnt_any_certified, cnt_all_certified = 0, 0, 0, 0, 0
    for x, y in data_loader:
        image, full_label = get_x_and_y(x, y, device, data_manager, params.skip, get_y=False, get_full_y=True)
        assert image.shape[0] == 1
        sim_images, sim_labels = data_loader.dataset.get_similar(full_label, params.perturb_factor)

        predictions = []
        certified = []
        for sim_image in sim_images:
            sim_z_center, r1, _ = center_smoothing.certify(
                sim_image.to(device), params.perturb_epsilon,
                params.sampling_batch_size
            )
            is_certified = False
            if sim_z_center is None:
                assert r1 == CenterSmoothing.ABSTAIN
                p = -1
            else:
                p, r2 = classifier_smoothing.certify(
                    sim_z_center.unsqueeze(0), params.cls_n0, params.cls_n,
                    params.cls_alpha, params.sampling_batch_size
                )
                if r1 < r2:
                    is_certified = True
            certified.append(is_certified)
            predictions.append(p)

        cnt_emp_unfair += (len(set(predictions)) > 1)
        cnt_any_certified_but_unfair += (any(certified) and len(set(predictions)) > 1)
        cnt_any_certified += any(certified)
        cnt_all_certified += all(certified)
        cnt_total += 1
        logger.debug('[n = {}] sample_idx = {}, {}, unfair = {:.1f}%, (any) certified but unfair = {:.1f}%'.format(
            cnt_total, sample_idx, Counter(predictions),
            100 * cnt_emp_unfair / cnt_total,
            100 * cnt_any_certified_but_unfair / cnt_total
        ))
        logger.debug('any certified = {:.1f}%, all certified = {:.1f}%'.format(
            100 * cnt_any_certified / cnt_total,
            100 * cnt_all_certified / cnt_total
        ))
        logger.debug(f'Pred = {predictions}')
        logger.debug(f'Cert = {[int(c) for c in certified]}')

        if params.save_reconstruction_images:
            rec_images = []
            with torch.no_grad():
                z_gen_model_latents = gen_model.latents(sim_images[:1].to(device))

                for v in torch.linspace(0, 1, factor_values):
                    z_gen_model_latents_cur = [
                        z_i + v * z_delta_i
                        for z_i, z_delta_i in zip(z_gen_model_latents, attr_vector)
                    ]
                    rec_images.append(
                        gen_model.reconstruct_from_latents(z_gen_model_latents_cur)
                    )
            rec_images = torch.cat(rec_images).cpu()

            fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(factor_values, 2))
            ax[0].imshow(
                make_grid(sim_images, nrow=factor_values, normalize=True).permute(1, 2, 0)
            )
            ax[1].imshow(
                make_grid(rec_images, nrow=factor_values, normalize=True).permute(1, 2, 0)
            )
            for idx in range(2):
                ax[idx].set_xticks(list())
                ax[idx].set_yticks(list())
                ax[idx].grid(False)

            fig.tight_layout()
            plt.savefig(attr_vectors_dir / f'{sample_idx}.eps', )
            plt.close(fig)

        sample_idx += params.skip


if __name__ == '__main__':
    params = get_params()
    utils.common_experiment_setup(params.seed)
    main(params)
