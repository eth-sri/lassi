# Adapted from the original center smoothing paper:
# Center Smoothing: Provable Robustness for Functions with Metric-Space Outputs
# Accompanying source code: https://github.com/aounon/center-smoothing/blob/main/center_smoothing.py

import argparse
import math
from typing import Optional

import numpy as np
from scipy.stats import norm
import torch

from models.generative_model import GenerativeModel
from models.gen_model_factory import GenModelWrapper
from models.lassi_encoder import LASSIEncoder
from sampler import GaussianNoiseAdder


def repeat_along_dim(t: torch.tensor, num_repeat: int, dim: int = 0) -> torch.tensor:
    return torch.repeat_interleave(torch.unsqueeze(t, dim), num_repeat, dim=dim)


def l2_dist(batch1, batch2):
    dist = torch.linalg.norm(batch1 - batch2, ord=2, dim=1)
    dist = dist.cpu().numpy()
    return dist


class CenterSmoothing:

    ABSTAIN = -1.0

    # Keep the default arguments.
    def __init__(self, params: argparse.Namespace, gen_model: Optional[GenerativeModel], lassi_encoder: LASSIEncoder,
                 sigma: float, dist_fn=l2_dist, n_pred: int = 10 ** 4, n_cert: int = 10 ** 6, n_cntr: int = 30,
                 alpha_1: float = 0.005, alpha_2: float = 0.005, triang_delta: float = 0.05,
                 radius_coeff: int = 3, output_is_hd: bool = False):
        self.gen_model_wrapper = GenModelWrapper(params, gen_model)
        self.lassi_encoder = lassi_encoder
        self.noise_adder = GaussianNoiseAdder(sigma)

        self.dist_fn = dist_fn
        self.sigma = sigma
        self.n_pred = n_pred    # Number of samples used for prediction (center computation)
        self.n_cert = n_cert    # Number of samples used for certification
        self.n_cntr = n_cntr    # Number of candidate centers
        self.alpha_1 = alpha_1
        self.alpha_2 = alpha_2
        self.triang_delta = triang_delta
        self.radius_coeff = radius_coeff    # for the relaxed triangle inequality
        self.output_is_hd = output_is_hd    # whether to use procedure for high-dimensional outputs

    def certify(self, single_x: torch.tensor, eps_in: float, batch_size: int = 1000):
        with torch.no_grad():
            # Computing center
            if self.output_is_hd:
                center, is_good = self.compute_center_hd(single_x, batch_size=batch_size)
            else:
                center, is_good = self.compute_center(single_x, batch_size=batch_size)

            if not is_good:
                return None, CenterSmoothing.ABSTAIN, CenterSmoothing.ABSTAIN

            # Calculating smoothing error
            z_gen_model_wrapped_latents_batch = self.gen_model_wrapper.get_wrapped_latents(single_x.unsqueeze(0))
            model_output = self.base_function(z_gen_model_wrapped_latents_batch, add_noise=False)
            assert model_output.dim() == 2
            assert model_output.size(0) == 1 and model_output.size(1) == self.lassi_encoder.latent_dimension()
            assert center.dim() == 1 and center.size(0) == self.lassi_encoder.latent_dimension()
            smoothing_error = self.dist_fn(center.unsqueeze(0), model_output)

            # Computing certificate
            min_prob = 0.5 + self.triang_delta
            quantile = norm.cdf(norm.ppf(min_prob) + (eps_in / self.sigma)) + \
                       math.sqrt(math.log(1 / self.alpha_2) / (2 * self.n_cert))

            if quantile > 1.0 or quantile < 0.0:
                print('Invalid quantile value: %.3f' % quantile)
                print(norm.cdf(norm.ppf(min_prob) + (eps_in / self.sigma)))
                print(math.sqrt(math.log(1 / self.alpha_2) / (2 * self.n_cert)))
                return center, CenterSmoothing.ABSTAIN, smoothing_error

            dist = np.zeros(self.n_cert)
            num = self.n_cert
            start = 0

            single_z_gen_model_wrapped_latents = z_gen_model_wrapped_latents_batch.squeeze(0)
            batch_inp_z_gen_model_wrapped_latents = repeat_along_dim(single_z_gen_model_wrapped_latents, batch_size)
            batch_cen = repeat_along_dim(center, batch_size)

            while num > 0:
                this_batch_size = min(batch_size, num)
                num -= this_batch_size

                if this_batch_size != batch_size:
                    batch_inp_z_gen_model_wrapped_latents = \
                        repeat_along_dim(single_z_gen_model_wrapped_latents, this_batch_size)
                    batch_cen = repeat_along_dim(center, this_batch_size)

                samples = self.base_function(batch_inp_z_gen_model_wrapped_latents, add_noise=True)

                dist_batch = self.dist_fn(samples, batch_cen)
                end = start + this_batch_size
                dist[start:end] = dist_batch
                start = end

            eps_out = self.radius_coeff * np.quantile(dist, quantile)   # 3 * np.quantile(dist, quantile)

            return center, eps_out, smoothing_error

    def compute_center(self, single_x: torch.tensor, batch_size: int = 1000):
        # Smoothing procedure
        with torch.no_grad():
            delta_1 = math.sqrt(math.log(2 / self.alpha_1) / (2 * self.n_pred))
            is_good = False
            num = self.n_pred

            single_z_gen_model_wrapped_latents = self.gen_model_wrapper.get_wrapped_latents(
                single_x.unsqueeze(0)
            ).squeeze(0)
            batch_inp_z_gen_model_wrapped_latents = repeat_along_dim(single_z_gen_model_wrapped_latents, batch_size)
            samples = None

            while num > 0:
                this_batch_size = min(batch_size, num)
                num -= this_batch_size

                if this_batch_size != batch_size:
                    batch_inp_z_gen_model_wrapped_latents = \
                        repeat_along_dim(single_z_gen_model_wrapped_latents, this_batch_size)

                z_batch = self.base_function(batch_inp_z_gen_model_wrapped_latents, add_noise=True)

                if samples is None:
                    samples = z_batch
                else:
                    samples = torch.cat((samples, z_batch), 0)

            center, radius = self.meb(samples)
            num_pts = self.pts_in_nbd(single_z_gen_model_wrapped_latents, center, radius, batch_size=batch_size)

            frac = num_pts / self.n_pred
            p_delta_1 = frac - delta_1
            delta_2 = (1 / 2) - p_delta_1

            # print(max(delta_1, delta_2))
            if max(delta_1, delta_2) <= self.triang_delta:
                is_good = True
                # print('Good center.')
            else:
                # print('Bad center. Abstaining ...')
                pass

        return center, is_good

    def compute_center_hd(self, input: torch.tensor, batch_size: int = 1000):
        raise NotImplementedError('`CenterSmoothing.compute_center_hd` is not currently used')

    def pts_in_nbd(self, single_z_gen_model_wrapped_latents: torch.tensor, center: torch.tensor, radius: float,
                   batch_size: int = 1000):
        with torch.no_grad():
            num = self.n_pred
            num_pts = 0

            batch_inp_z_gen_model_wrapped_latents = repeat_along_dim(single_z_gen_model_wrapped_latents, batch_size)
            batch_cen = repeat_along_dim(center, batch_size)

            while num > 0:
                this_batch_size = min(batch_size, num)
                num -= this_batch_size

                if this_batch_size != batch_size:
                    batch_inp_z_gen_model_wrapped_latents = \
                        repeat_along_dim(single_z_gen_model_wrapped_latents, this_batch_size)
                    batch_cen = repeat_along_dim(center, this_batch_size)

                samples = self.base_function(batch_inp_z_gen_model_wrapped_latents, add_noise=True)

                dist = self.dist_fn(samples, batch_cen)
                num_pts += np.sum(np.where(dist <= radius, 1, 0))
        return num_pts

    def meb(self, samples):
        with torch.no_grad():
            radius = np.inf
            num_samples = samples.shape[0]
            for i in range(num_samples):
                curr_sample = samples[i]
                sample_batch = repeat_along_dim(curr_sample, num_samples)
                dist = self.dist_fn(samples, sample_batch)
                median_dist = np.median(dist)
                if median_dist < radius:
                    radius = median_dist
                    center = curr_sample

        return center, radius

    def base_function(self, batch_inp_z_gen_model_wrapped_latents: torch.tensor, add_noise: bool = True) \
            -> torch.tensor:
        if add_noise:
            z_gen_model_wrapped_latents_batch = self.noise_adder.add_noise(
                self.gen_model_wrapper.gen_model,
                batch_inp_z_gen_model_wrapped_latents
            )
        else:
            z_gen_model_wrapped_latents_batch = batch_inp_z_gen_model_wrapped_latents

        enc_input = self.gen_model_wrapper.compute_encoder_input(z_gen_model_wrapped_latents_batch)
        return self.lassi_encoder(enc_input)
