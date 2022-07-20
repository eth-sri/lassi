import argparse
import itertools
from collections import defaultdict
from itertools import combinations
import math
from pathlib import Path
from typing import List

from loguru import logger
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

import args
from dataset.data_manager import DataManager
from dataset.fairface_dataset import FairFaceDataset
from metrics import ClassifierMetrics
from models.classifiers import OriginalImagesClassifier
from models.gen_model_factory import GenModelFactory
from models.glow import Glow
from standard_classifier.train_model import StandardClassifierTrainer
import utils


AVERAGE_DIFF = 'average_diff'
PERPENDICULAR = 'perpendicular'  # Perpendicular to the decision boundary (Denton et al.).
RAMASWAMY = 'ramaswamy'          # Ramaswamy et al. (extension of Denton et al.).
DISCOVER = 'discover'            # Discover the Unknown Biased Attribute of an Image Classifier (Li and Xu).
METHOD_TO_ATTR_VECTORS_FOLDER_NAME = {
    AVERAGE_DIFF:  'attr_vectors_avg_diff',
    PERPENDICULAR: 'attr_vectors_perp',
    RAMASWAMY:     'attr_vectors_ram',
    DISCOVER:      'attr_vectors_discover'
}


def get_params() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute the (semantic) attribute vectors of the pretrained Glow model",
        parents=[
            args.common(),
            args.dataset(),
            args.gen_model(),
            args.glow(),
            args.encoder()
        ],
        conflict_handler='resolve'
    )

    parser.add_argument('--computation_method', type=str, default=AVERAGE_DIFF,
                        choices=[AVERAGE_DIFF, PERPENDICULAR, RAMASWAMY, DISCOVER],
                        help='Computation method.')
    parser.add_argument('--target', type=str, default='Smiling', help="Target attribute (needed for Ramaswamy et al.)")
    parser.add_argument('--attributes', type=args.argslist_str, default=[],
                        help="For which sensitive attributes to compute the semantic (attribute) vectors")
    parser.add_argument('--rewrite', type=args.bool_flag, default=False,
                        help="If a vector already exists, whether to rewrite it or not")
    parser.add_argument('--normalize_vectors', type=args.bool_flag, default=False,
                        help="Normalize the attribute vectors")

    # (Default) Li and Xu method arguments.
    parser.add_argument('--fair_classifier_name', type=str,
                        help="Path to the standard (normally pretrained, biased) classifier")
    parser.add_argument('--num_biased_attributes', type=int, default=3,
                        help="How many (biased) sensitive attribute vectors to discover")
    parser.add_argument('--discover_attrib_num_iters', type=int, default=2000,
                        help="For how many iterations to run the optimization to discover a single biased attribute")
    parser.add_argument('--discover_attrib_batch_size', type=int, default=4,
                        help="Discover a single biased attribute - optimization batch size")
    parser.add_argument('--discover_attrib_lr', type=float, default=1e-3,
                        help="Learning rate of the optimizer used for discovering the unknown biased attributes")
    parser.add_argument('--lambda_orth_coeff', type=float, default=10.0,
                        help="Lambda coefficient of the orthogonality penalty")
    parser.add_argument('--traverse_min', default=-5, type=float, help="Min limit to traverse latents")
    parser.add_argument('--traverse_max', default=+5, type=float, help="Max limit to traverse latents")
    parser.add_argument('--traverse_spacing', default=5, type=float, help="Spacing to traverse latents")

    params = parser.parse_args()

    if 'fairface' not in params.dataset:
        assert not params.classify_attributes
        # To make it easier to work with the DataManager:
        if params.computation_method != DISCOVER:
            params.classify_attributes = params.attributes
        if params.computation_method in [RAMASWAMY, DISCOVER]:
            params.classify_attributes.append(params.target)
    else:
        assert params.computation_method == AVERAGE_DIFF

    return params


def get_z_zero(glow: Glow, device) -> List[torch.Tensor]:
    z_list = []
    for z_shape in glow.z_shapes:
        z_list.append(torch.zeros(*z_shape).to(device))
    return z_list


def prepare(params: argparse.Namespace):
    device = utils.get_device(params.use_cuda)
    data_manager = DataManager.get_manager(params)
    glow: Glow = GenModelFactory.get_and_load(params, freeze=True).to(device)
    file_or_dir = utils.get_path_to('saved_models') / params.gen_model_name
    attr_vectors_folder_name = METHOD_TO_ATTR_VECTORS_FOLDER_NAME[params.computation_method]
    attr_vectors_dir = utils.get_or_create_path(utils.get_directory_of_path(file_or_dir) / attr_vectors_folder_name)
    return device, data_manager, glow, attr_vectors_dir


def write_attribute_vector(attr_vectors_dir: Path, attr_name: str, attr_vector_list, normalize=True, rewrite=False):
    l2_norm_squared = 0.0
    for z_i in attr_vector_list:
        l2_norm_squared += (z_i * z_i).sum()
    l2_norm = math.sqrt(l2_norm_squared)
    logger.debug(f'Original l2 norm: {l2_norm:.5f}')
    if normalize:
        logger.debug('Normalize vector')
        attr_vector_list = [z_i / l2_norm for z_i in attr_vector_list]
    attr_vector_file = attr_vectors_dir / f'{attr_name}.npz'
    if not attr_vector_file.is_file() or rewrite:
        logger.debug(f"Saving {attr_vector_file}...")
        np.savez(attr_vector_file, *attr_vector_list)


def has_attribute_mask(y: torch.Tensor, attr_index: int, params: argparse.Namespace) -> torch.Tensor:
    if 'celeba' in params.dataset:
        yi = y[:, attr_index]
        assert yi.dim() == 1 and yi.size(0) == y.size(0)
        return yi
    elif 'fairface' in params.dataset:
        assert len(params.classify_attributes) == 1 and params.classify_attributes[0] == 'Race', \
            'Computing attribute mask for FairFace/Age is not implemented.'
        race_idx = FairFaceDataset.FAIR_FACE_RACES.index(params.attributes[attr_index])
        return (y == race_idx).long()
    else:
        raise ValueError(f'Dataset {params.dataset} not supported')


def compute_attribute_vectors_avg_diff(params: argparse.Namespace):
    assert params.gen_model_type == 'Glow'
    device, data_manager, glow, attr_vectors_dir = prepare(params)

    logger.debug(f'Computation method: average difference (Glow paper)')
    logger.debug(f'Will save the attribute vectors at dir: {attr_vectors_dir}')
    logger.debug(f'Sensitive attributes: {params.attributes}')

    num_attributes = len(params.attributes)
    cnt_pos, cnt_neg = [0] * num_attributes, [0] * num_attributes
    z_pos_per_attribute = [get_z_zero(glow, device) for _ in params.attributes]
    z_neg_per_attribute = [get_z_zero(glow, device) for _ in params.attributes]

    for x, y in tqdm(data_manager.get_dataloader('train', shuffle=False), ncols=100):
        y = data_manager.target_transform(y)
        x = x.to(device)
        y = y.to(device)
        cur_batch_size = x.size(0)

        z_outs = glow.unwrap_latents(x)

        for attr_idx in range(num_attributes):
            pos_mask = has_attribute_mask(y, attr_idx, params)
            neg_mask = 1 - pos_mask

            a = pos_mask.count_nonzero().item()
            b = cur_batch_size - a
            cnt_pos[attr_idx] += a
            cnt_neg[attr_idx] += b

            for i, z_out in enumerate(z_outs):
                z_pos = z_out * pos_mask.view(cur_batch_size, 1, 1, 1)
                z_neg = z_out * neg_mask.view(cur_batch_size, 1, 1, 1)

                z_pos_per_attribute[attr_idx][i] += z_pos.sum(0)
                z_neg_per_attribute[attr_idx][i] += z_neg.sum(0)

    for p, n, attr_name in zip(cnt_pos, cnt_neg, params.attributes):
        logger.debug('{:>15}: pos = {:>5}, neg = {:>5}'.format(attr_name, p, n))

    for attr_idx, attr_name in enumerate(params.attributes):
        z_pos_list = z_pos_per_attribute[attr_idx]
        z_neg_list = z_neg_per_attribute[attr_idx]

        attr_vector_list = [((z_pos / cnt_pos[attr_idx]) - (z_neg / cnt_neg[attr_idx])).detach().cpu().numpy()
                            for z_pos, z_neg in zip(z_pos_list, z_neg_list)]
        write_attribute_vector(
            attr_vectors_dir, attr_name, attr_vector_list,
            normalize=params.normalize_vectors, rewrite=params.rewrite
        )


def compute_attribute_vectors_avg_diff_all_pairs(params: argparse.Namespace):
    assert params.gen_model_type == 'Glow'
    device, data_manager, glow, attr_vectors_dir = prepare(params)

    logger.debug(f'Computation method: average difference (Glow paper) all pairs')
    logger.debug(f'Will save the attribute vectors at dir: {attr_vectors_dir}')
    logger.debug(f'Sensitive attributes: {params.attributes}')

    num_attributes = len(params.attributes)
    counts = defaultdict(lambda: [0] * num_attributes)
    z_per_attribute = defaultdict(lambda: [get_z_zero(glow, device) for _ in params.attributes])

    for x, y in tqdm(data_manager.get_dataloader('train', shuffle=False), ncols=100):
        y = data_manager.target_transform(y)
        if y.dim() == 1:
            y = y.unsqueeze(dim=1)
        x = x.to(device)
        y = y.to(device)

        for i in range(num_attributes):
            yi = y[:, i]
            assert yi.dim() == 1 and yi.size(0) == y.size(0)
            for val, count in zip(*torch.unique(yi, return_counts=True)):
                counts[val.item()][i] += count.item()

        z_outs = glow.unwrap_latents(x)

        for attr_idx in range(num_attributes):
            yi = y[:, attr_idx]
            assert yi.dim() == 1 and yi.size(0) == y.size(0)

            for i, z_out in enumerate(z_outs):
                for val in torch.unique(yi, sorted=True):
                    z_per_attribute[val.item()][attr_idx][i] += torch.sum(z_out[yi == val], dim=0)

    for attr_idx, attr_name in enumerate(params.attributes):
        logger.debug(
            f'{attr_name:>15}: ' + ', '.join([
                f'{key}={count[attr_idx]:>5}' for key, count in counts.items()
            ])
        )

    for attr_idx, attr_name in enumerate(params.attributes):
        for beg, end in combinations(sorted(z_per_attribute.keys()), r=2):
            counts_beg = counts[beg][attr_idx]
            counts_end = counts[end][attr_idx]
            if counts_beg == 0 or counts_end == 0:
                continue
            z_beg_list = z_per_attribute[beg][attr_idx]
            z_end_list = z_per_attribute[end][attr_idx]

            attr_vector_list = [
                torch.detach(
                    (z_end / counts_end) - (z_beg / counts_beg)
                ).cpu().numpy() for z_beg, z_end in zip(z_beg_list, z_end_list)
            ]

            write_attribute_vector(
                attr_vectors_dir, f'{attr_name}_{beg}_{end}', attr_vector_list,
                normalize=params.normalize_vectors, rewrite=params.rewrite
            )


def train_linear_cls(params: argparse.Namespace, data_manager: DataManager, glow: Glow, device, attr_idx: int):
    # attr_idx is an index of params.classify_attributes
    linear_cls = nn.Linear(glow.latent_dimension(), 1).to(device)
    loss_fn = nn.BCEWithLogitsLoss()
    opt = optim.Adam(linear_cls.parameters(), lr=params.lr)
    best_loss, best_epoch, coeffs, bias = float('inf'), -1, None, None

    def train_loop(split: str):
        assert split in ['train', 'valid']
        total_loss = 0.0
        total_samples = 0
        metrics = ClassifierMetrics(split)
        for x, y in tqdm(data_manager.get_dataloader(split, shuffle=(split == 'train')), ncols=100, leave=False):
            y = data_manager.target_transform(y)
            if y.dim() == 1:
                y = y.unsqueeze(dim=1)
            y = y[:, attr_idx]

            x = x.to(device)
            y = y.to(device)
            total_samples += x.size(0)

            logits = linear_cls(x)
            loss = loss_fn(logits.squeeze(1), y.float())
            total_loss += loss.item() * x.size(0)

            if split == 'train':
                opt.zero_grad()
                loss.backward()
                opt.step()

            y_pred = (logits.squeeze(1) > 0)
            metrics.add(batch_loss=loss.item(), y_targets=y, y_pred=y_pred)
        return total_loss / total_samples, metrics.get_accuracies()[0]

    for epoch in range(params.epochs):
        linear_cls.train()
        train_loop('train')
        linear_cls.eval()
        with torch.no_grad():
            valid_loss, valid_acc = train_loop('valid')
            logger.debug(f'[Epoch = {epoch}] valid_loss = {valid_loss:.5f}, valid_acc = {valid_acc:.2f}')
            if valid_loss < best_loss:
                best_loss = valid_loss
                best_epoch = epoch
                coeffs = linear_cls.weight.detach().clone().cpu()
                bias = linear_cls.bias.item()

    logger.debug(f'Best epoch: {best_epoch}')
    return coeffs, bias


def compute_attribute_vectors_perpendicular(params: argparse.Namespace):
    assert params.gen_model_type == 'Glow'
    utils.set_random_seed(params.seed)

    device, data_manager, glow, attr_vectors_dir = prepare(params)

    logger.debug(f'Computation method: perpendicular (Denton et al.)')
    logger.debug(f'Will save the attribute vectors at dir: {attr_vectors_dir}')
    logger.debug(f'Sensitive attributes: {params.attributes}')

    for attr_idx, attr_name in enumerate(params.attributes):
        logger.debug(f'Compute attribute vector for attribute: {attr_name}')
        coeffs, _ = train_linear_cls(params, data_manager, glow, device, attr_idx)
        assert coeffs.dim() == 2 and coeffs.size(0) == 1 and coeffs.size(1) == glow.latent_dimension()
        attr_vector_list = glow.unwrap_latents(coeffs)
        attr_vector_list = [a.squeeze(dim=0) for a in attr_vector_list]
        write_attribute_vector(
            attr_vectors_dir, attr_name, attr_vector_list,
            normalize=params.normalize_vectors, rewrite=params.rewrite
        )


def compute_attribute_vectors_ramaswamy(params: argparse.Namespace):
    assert params.gen_model_type == 'Glow'
    utils.set_random_seed(params.seed)

    device, data_manager, glow, attr_vectors_dir = prepare(params)

    logger.debug(f'Computation method: Ramaswamy et al.')
    logger.debug(f'Will save the attribute vectors at dir: {attr_vectors_dir}')
    logger.debug(f'Target: {params.target}')
    logger.debug(f'Sensitive attributes: {params.attributes}')

    ws, bs = [], []

    for attr_idx, attr_name in enumerate(params.classify_attributes):
        logger.debug(f'Train linear boundary for attribute: {attr_name}')
        w, b = train_linear_cls(params, data_manager, glow, device, attr_idx)
        assert w.dim() == 2 and w.size(0) == 1 and w.size(1) == glow.latent_dimension()
        w = w.squeeze(dim=0)
        w_norm = torch.linalg.norm(w, ord=2).item()
        ws.append((w / w_norm).to(device))
        bs.append(b / w_norm)

    assert params.classify_attributes[-1] == params.target
    w_target = ws[-1]

    a_dir, a_dir_scaled = [], []
    for w_g in ws[:-1]:
        dot_product = torch.dot(w_g, w_target)
        a_dir.append(w_g - dot_product * w_target)
        a_dir_scaled.append(2 * a_dir[-1] / (1 - dot_product * dot_product))

    distances_distribution = [[] for _ in a_dir_scaled]
    distances_sum = [0.0] * len(params.attributes)
    total_samples = 0

    for x, _ in tqdm(data_manager.get_dataloader('train', shuffle=False), ncols=100):
        z = x.to(device)
        batch_size = z.size(0)
        total_samples += batch_size
        for i, w_g, b_g, a in zip(itertools.count(), ws[:-1], bs[:-1], a_dir_scaled):
            sample_specific_scale_coeff = (z * w_g).sum(dim=1) + b_g
            a_repeated = torch.repeat_interleave(a.unsqueeze(0), batch_size, dim=0)
            z_deltas = sample_specific_scale_coeff.unsqueeze(1) * a_repeated
            assert z.shape == z_deltas.shape
            distances = torch.linalg.norm(z_deltas, ord=2, dim=1)
            distances_sum[i] += distances.sum().item()
            distances_distribution[i].extend(distances.cpu().tolist())

    for a, d_sum, attr_name in zip(a_dir, distances_sum, params.attributes):
        avg_d = d_sum / total_samples
        a_norm = torch.linalg.norm(a, ord=2).item()
        a_unit = a / a_norm
        a_avg_scaled = a_unit * avg_d

        attrib_vec = a_avg_scaled.cpu().unsqueeze(0)
        assert attrib_vec.dim() == 2 and attrib_vec.size(0) == 1 and attrib_vec.size(1) == glow.latent_dimension()
        attr_vector_list = glow.unwrap_latents(attrib_vec)
        attr_vector_list = [a.squeeze(dim=0) for a in attr_vector_list]
        write_attribute_vector(
            attr_vectors_dir, attr_name, attr_vector_list,
            normalize=False, rewrite=params.rewrite
        )


class DiscoverUnknownBiasTrainer:
    def __init__(self, params: argparse.Namespace, device, data_manager: DataManager, glow: Glow,
                 std_cls_model: OriginalImagesClassifier, ws_known: List[torch.Tensor]):
        self.device = device
        self.data_manager = data_manager
        self.glow = glow
        self.std_cls_model = std_cls_model
        self.num_iters = params.discover_attrib_num_iters
        self.batch_size = params.discover_attrib_batch_size

        w = torch.randn(1, self.glow.latent_dimension(), device=self.device)
        b = torch.randn(1, 1, device=self.device)

        self.w = torch.nn.Parameter(w, requires_grad=True)
        self.b = torch.nn.Parameter(b, requires_grad=True)
        self.optimizer = torch.optim.Adam([self.w, self.b], params.discover_attrib_lr)

        ws_known_ma3x = torch.cat(ws_known, dim=0).t()  # The w vectors are in the columns.
        other_unit_normal_vectors = ws_known_ma3x / torch.linalg.norm(ws_known_ma3x, ord=2, dim=0).unsqueeze(0)
        self.other_unit_normal_vectors = other_unit_normal_vectors.to(self.device)
        self.other_unit_normal_vectors_row_ma3x = self.other_unit_normal_vectors.t()

        self.lambda_coeff = params.lambda_orth_coeff
        self.traverse_steps = torch.arange(
            params.traverse_min, params.traverse_max + 0.001, params.traverse_spacing,
            device=self.device
        ).reshape(1, -1, 1)

    @staticmethod
    def project_point_to_plane(points, normal_vec, offset):
        projected_proints = points - normal_vec * (points @ normal_vec.t() + offset / (normal_vec @ normal_vec.T))
        return projected_proints

    @staticmethod
    def traverse_along_normal_vec(projected_points, normal_vector, traverse_steps, direction_vector=None):
        projected_points = projected_points.unsqueeze(1)
        if direction_vector is None:
            unit_normal_vec = normal_vector / torch.norm(normal_vector)
            unit_normal_vec = unit_normal_vec.unsqueeze(1)
            traverse_points = projected_points + unit_normal_vec * traverse_steps
        else:
            unit_direction_vec = direction_vector / torch.norm(direction_vector)
            traverse_points = projected_points + unit_direction_vec / \
                              (direction_vector @ normal_vector.t()) * traverse_steps
        return traverse_points

    def train(self):
        total_loss = 0
        total_tv_loss = 0
        total_orthogonal_constraint_loss = 0

        # pbar = tqdm(range(self.num_iters), dynamic_ncols=True)
        pbar = tqdm(
            self.data_manager.get_dataloader('train', shuffle=True, batch_size=self.batch_size),
            dynamic_ncols=True
        )
        idx = 0
        for x, _ in pbar:
            # z = torch.rand(self.batch_size, self.glow.latent_dimension(), device=self.device)
            z = x.to(self.device)
            idx += 1

            z_projected = self.project_point_to_plane(z, self.w, self.b)
            z_traverse = self.traverse_along_normal_vec(z_projected, self.w, self.traverse_steps)

            z_traverse_latents = z_traverse.reshape(-1, self.glow.latent_dimension())
            z_traverse_latents_unwrapped = self.glow.unwrap_latents(z_traverse_latents)
            out = self.glow.reconstruct_from_latents(z_traverse_latents_unwrapped)

            logits = self.std_cls_model(out)
            probs = F.softmax(logits, dim=1)[:, 1]
            probs = probs.reshape(self.batch_size, self.traverse_steps.shape[1])

            # Total variation loss:
            tv_loss = -torch.log(1e-10 + torch.abs(probs[:, 1:] - probs[:, :-1]).mean())
            total_tv_loss += tv_loss.item()
            # Orthogonality penalty:
            orthogonal_constraint_loss = (
                ((self.w / torch.norm(self.w)) @ self.other_unit_normal_vectors) ** 2
            ).mean()
            total_orthogonal_constraint_loss += orthogonal_constraint_loss.item()
            # Total (combined) loss:
            loss = tv_loss + self.lambda_coeff * orthogonal_constraint_loss
            total_loss += loss.item()

            # Backpropagate the total loss.
            self.optimizer.zero_grad()
            self.std_cls_model.zero_grad()
            self.glow.zero_grad()
            loss.backward()
            self.optimizer.step()

            with torch.no_grad():
                cosine_similarities = []
                for w_i in self.other_unit_normal_vectors_row_ma3x:
                    cos_sim = F.cosine_similarity(self.w, w_i.unsqueeze(0))
                    cosine_similarities.append(f'{cos_sim.item():.3f}')

            pbar.set_postfix(
                tot_loss=loss.item(),  # total_loss / idx,
                avg_tot_loss=total_loss / idx,
                tv_loss=tv_loss.item(),  # total_tv_loss / idx,
                ort_loss=orthogonal_constraint_loss.item(),  # total_orthogonal_constraint_loss / idx,
                cos_sim='[' + ','.join(cosine_similarities) + ']'
            )
            pbar.update()

            if idx >= self.num_iters:
                break


def compute_attribute_vectors_discover(params: argparse.Namespace):
    assert params.gen_model_type == 'Glow'
    utils.set_random_seed(params.seed)

    device, data_manager, glow, attr_vectors_dir = prepare(params)
    assert params.fair_classifier_name is not None
    std_cls_model = StandardClassifierTrainer.load_models(params).to(device)

    logger.debug(f'Computation method: Discover attribute vectors (Li and Xu)')
    logger.debug(f'Will save the attribute vectors at dir: {attr_vectors_dir}')
    logger.debug(f'Target: {params.target}')

    logger.debug(f'Train linear boundary for attribute: {params.target}')
    w, _ = train_linear_cls(params, data_manager, glow, device, 0)
    assert w.dim() == 2 and w.size(0) == 1 and w.size(1) == glow.latent_dimension()

    ws_known = [w]
    for i in range(params.num_biased_attributes):
        trainer = DiscoverUnknownBiasTrainer(params, device, data_manager, glow, std_cls_model, ws_known)
        trainer.train()

        attrib_vec = trainer.w.detach().cpu()
        ws_known.append(attrib_vec)

        assert attrib_vec.dim() == 2 and attrib_vec.size(0) == 1 and attrib_vec.size(1) == glow.latent_dimension()
        attr_vector_list = glow.unwrap_latents(attrib_vec)
        attr_vector_list = [a.squeeze(dim=0) for a in attr_vector_list]
        write_attribute_vector(
            attr_vectors_dir, f'biased_attr_{i}', attr_vector_list,
            normalize=True, rewrite=params.rewrite
        )


if __name__ == '__main__':
    p = get_params()
    assert p.dataset.endswith('_latent_lmdb')
    utils.init_logger()
    if p.computation_method == AVERAGE_DIFF:
        if '3dshapes' in p.dataset:
            compute_attribute_vectors_avg_diff_all_pairs(p)
        else:
            compute_attribute_vectors_avg_diff(p)
    elif p.computation_method == PERPENDICULAR:
        assert 'celeba' in p.dataset
        compute_attribute_vectors_perpendicular(p)
    elif p.computation_method == RAMASWAMY:
        assert 'celeba' in p.dataset
        compute_attribute_vectors_ramaswamy(p)
    elif p.computation_method == DISCOVER:
        assert 'celeba' in p.dataset
        compute_attribute_vectors_discover(p)
    else:
        raise ValueError(f'The computation method {p.computation_method} is not supported.')
