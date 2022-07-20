import argparse
from typing import List, Tuple

from loguru import logger
import torch

from models.gen_model_factory import GenModelWrapper
from models.lassi_encoder import LASSIEncoder
from sampler import UniformNoiseAdder, UniformNoise2DAdder, UniformNoiseNDAdder


class Attack:

    def __init__(self, params: argparse.Namespace, gen_model_wrapper: GenModelWrapper, lassi_encoder:LASSIEncoder):
        self.gen_model_wrapper = gen_model_wrapper
        self.lassi_encoder = lassi_encoder
        self.delta = params.delta

    def compute_z_lassi(self, z_gen_model_latents_wrapped: torch.Tensor):
        enc_input = self.gen_model_wrapper.compute_encoder_input(z_gen_model_latents_wrapped)
        return self.lassi_encoder(enc_input)

    def calc_loss(self, z_gen_model_latents_wrapped: torch.Tensor, z_lassi: torch.Tensor):
        z_lassi_adv = self.compute_z_lassi(z_gen_model_latents_wrapped)
        l_2 = torch.linalg.norm(z_lassi - z_lassi_adv, ord=2, dim=1)
        return torch.clamp(l_2 - self.delta, min=0.0)


class RandomSamplingAttack(Attack):

    def __init__(self, params: argparse.Namespace, gen_model_wrapper: GenModelWrapper, lassi_encoder: LASSIEncoder):
        super(RandomSamplingAttack, self).__init__(params, gen_model_wrapper, lassi_encoder)

        assert 0 < len(params.perturb.split(','))
        assert params.perturb_epsilon is not None

        self.num_samples = params.random_attack_num_samples
        if len(params.perturb.split(',')) == 1:
            self.noise_adder = UniformNoiseAdder(params.perturb_epsilon)
        elif len(params.perturb.split(',')) == 2:
            self.noise_adder = UniformNoise2DAdder(params.perturb_epsilon)
        else:
            self.noise_adder = UniformNoiseNDAdder(params.perturb_epsilon)
        logger.debug(f'Use {self.noise_adder.__class__.__name__} for the RandomSamplingAttack')

    def get_adv_examples(self, z_gen_model_latents: List[torch.Tensor], z_lassi: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            z_gen_model_latents_all = []
            losses_all = []
            for _ in range(self.num_samples):
                z_gen_model_latents_noisy = self.noise_adder.add_noise(
                    self.gen_model_wrapper.gen_model,
                    self.gen_model_wrapper.gen_model.wrap_latents(z_gen_model_latents)
                )
                z_gen_model_latents_all.append(z_gen_model_latents_noisy.clone().detach())
                l = self.calc_loss(z_gen_model_latents_noisy, z_lassi)
                losses_all.append(l.clone().detach())
            losses_all = torch.stack(losses_all, dim=1)
            _, idx = torch.max(losses_all, dim=1)
            adv_examples = []
            for i, sample_idx in enumerate(idx.cpu().tolist()):
                adv_examples.append(z_gen_model_latents_all[sample_idx][i])
        return torch.stack(adv_examples, 0).clone().detach().requires_grad_(True)

    def augment_data(self, z_gen_model_latents: List[torch.Tensor], y: torch.Tensor) \
            -> Tuple[torch.Tensor, torch.Tensor]:
        assert y.dim() == 1
        y_repeated = y.repeat_interleave(self.num_samples)

        z_gen_model_latents_wrapped = self.gen_model_wrapper.gen_model.wrap_latents(z_gen_model_latents)
        assert z_gen_model_latents_wrapped.dim() == 2 and z_gen_model_latents_wrapped.size(0) == y.size(0)
        z_gen_model_latents_wrapped_repeated = z_gen_model_latents_wrapped.repeat_interleave(self.num_samples, dim=0)
        z_gen_model_latents_wrapped_repeated = self.noise_adder.add_noise(
            self.gen_model_wrapper.gen_model,
            z_gen_model_latents_wrapped_repeated
        )

        return self.compute_z_lassi(z_gen_model_latents_wrapped_repeated), y_repeated
