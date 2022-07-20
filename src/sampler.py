from abc import ABC, abstractmethod
import math

import torch

from models.generative_model import GenerativeModel


class NoiseAdder(ABC):

    def __init__(self):
        self.cur_batch_size = -1
        self.attribute_vectors_repeated = []

    def set_attribute_vectors_repeated(self, gen_model: GenerativeModel, required_batch_size: int, device, dtype):
        if self.cur_batch_size != required_batch_size:
            self.cur_batch_size = required_batch_size
            self.attribute_vectors_repeated = gen_model.get_attribute_vectors_wrapped_repeated(
                device, dtype, required_batch_size
            )

    def add_noise(self, gen_model: GenerativeModel, z_gen_model_latents_wrapped: torch.Tensor):
        self.set_attribute_vectors_repeated(
            gen_model,
            z_gen_model_latents_wrapped.size(0),
            z_gen_model_latents_wrapped.device,
            z_gen_model_latents_wrapped.dtype
        )
        return self._add_noise(z_gen_model_latents_wrapped)

    @abstractmethod
    def _add_noise(self, z_gen_model_latents_wrapped: torch.Tensor):
        pass


class GaussianNoiseAdder(NoiseAdder):

    def __init__(self, sigma: float):
        super(GaussianNoiseAdder, self).__init__()
        self.sigma = sigma

    def _add_noise(self, z_gen_model_latents_wrapped: torch.Tensor):
        noisy_latents = z_gen_model_latents_wrapped.clone().detach()
        for attr_vec_repeated in self.attribute_vectors_repeated:
            coeffs = torch.randn(self.cur_batch_size, 1, device=noisy_latents.device) * self.sigma
            noisy_latents += attr_vec_repeated * coeffs
        return noisy_latents


class UniformNoiseAdder(NoiseAdder):

    def __init__(self, perturb_epsilon: float):
        super(UniformNoiseAdder, self).__init__()
        self.perturb_epsilon = perturb_epsilon

    def _add_noise(self, z_gen_model_latents_wrapped: torch.Tensor):
        noisy_latents = z_gen_model_latents_wrapped.clone().detach()
        for attr_vec_repeated in self.attribute_vectors_repeated:
            coeffs = (2 * torch.rand(self.cur_batch_size, 1, device=noisy_latents.device) - 1) * self.perturb_epsilon
            noisy_latents += attr_vec_repeated * coeffs
        return noisy_latents


class UniformNoise2DAdder(NoiseAdder):

    def __init__(self, perturb_epsilon: float):
        super(UniformNoise2DAdder, self).__init__()
        self.perturb_epsilon = perturb_epsilon

    def _add_noise(self, z_gen_model_latents_wrapped: torch.Tensor):
        assert len(self.attribute_vectors_repeated) == 2
        a0 = self.attribute_vectors_repeated[0]
        a1 = self.attribute_vectors_repeated[1]
        device = z_gen_model_latents_wrapped.device

        theta = 2 * math.pi * torch.rand(self.cur_batch_size, device=device)
        r = self.perturb_epsilon * torch.sqrt(torch.rand(self.cur_batch_size, device=device))
        e0 = r * torch.cos(theta)
        e1 = r * torch.sin(theta)

        return z_gen_model_latents_wrapped + a0 * e0.view(self.cur_batch_size, 1) + a1 * e1.view(self.cur_batch_size, 1)


class UniformNoiseNDAdder(NoiseAdder):
    # http://extremelearning.com.au/how-to-generate-uniformly-random-points-on-n-spheres-and-n-balls/

    def __init__(self, perturb_epsilon: float):
        super(UniformNoiseNDAdder, self).__init__()
        self.perturb_epsilon = perturb_epsilon

    def _add_noise(self, z_gen_model_latents_wrapped: torch.Tensor):
        assert len(self.attribute_vectors_repeated) > 2

        n = len(self.attribute_vectors_repeated)
        device = z_gen_model_latents_wrapped.device

        u = torch.randn(self.cur_batch_size, n, device=device)
        norm = torch.linalg.norm(u, ord=2, dim=1)
        r = self.perturb_epsilon * torch.rand(self.cur_batch_size, device=device).pow(1.0 / n)
        e = u * r.view(-1, 1) / norm.view(-1, 1)

        ret = z_gen_model_latents_wrapped.clone().detach()
        for i, a_i in enumerate(self.attribute_vectors_repeated):
            e_i = e[:, i]
            ret += a_i * e_i.view(self.cur_batch_size, 1)
        return ret
