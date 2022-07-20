# Glow implementation adapted from https://github.com/rosinality/glow-pytorch.

import argparse
from math import log, pi, prod
from pathlib import Path
import statistics
from typing import List, Tuple

import numpy as np
from scipy import linalg as la
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.generative_model import GenerativeModel
import utils

logabs = lambda x: torch.log(torch.abs(x))


class ActNorm(nn.Module):
    def __init__(self, in_channel, logdet=True):
        super().__init__()

        self.loc = nn.Parameter(torch.zeros(1, in_channel, 1, 1))
        self.scale = nn.Parameter(torch.ones(1, in_channel, 1, 1))

        self.register_buffer("initialized", torch.tensor(0, dtype=torch.uint8))
        self.logdet = logdet

    def initialize(self, input):
        with torch.no_grad():
            flatten = input.permute(1, 0, 2, 3).contiguous().view(input.shape[1], -1)
            mean = (
                flatten.mean(1)
                .unsqueeze(1)
                .unsqueeze(2)
                .unsqueeze(3)
                .permute(1, 0, 2, 3)
            )
            std = (
                flatten.std(1)
                .unsqueeze(1)
                .unsqueeze(2)
                .unsqueeze(3)
                .permute(1, 0, 2, 3)
            )

            self.loc.data.copy_(-mean)
            self.scale.data.copy_(1 / (std + 1e-6))

    def forward(self, input):
        _, _, height, width = input.shape

        if self.initialized.item() == 0:
            self.initialize(input)
            self.initialized.fill_(1)

        log_abs = logabs(self.scale)
        logdet = height * width * torch.sum(log_abs)

        if self.logdet:
            return self.scale * (input + self.loc), logdet
        else:
            return self.scale * (input + self.loc)

    def reverse(self, output):
        return output / self.scale - self.loc


class InvConv2d(nn.Module):
    def __init__(self, in_channel):
        super().__init__()

        weight = torch.randn(in_channel, in_channel)
        q, _ = torch.qr(weight)
        weight = q.unsqueeze(2).unsqueeze(3)
        self.weight = nn.Parameter(weight)

    def forward(self, input):
        _, _, height, width = input.shape

        out = F.conv2d(input, self.weight)
        logdet = (
            height * width * torch.slogdet(self.weight.squeeze().double())[1].float()
        )

        return out, logdet

    def reverse(self, output):
        return F.conv2d(
            output, self.weight.squeeze().inverse().unsqueeze(2).unsqueeze(3)
        )


class InvConv2dLU(nn.Module):
    def __init__(self, in_channel):
        super().__init__()

        weight = np.random.randn(in_channel, in_channel)
        q, _ = la.qr(weight)
        w_p, w_l, w_u = la.lu(q.astype(np.float32))
        w_s = np.diag(w_u)
        w_u = np.triu(w_u, 1)
        u_mask = np.triu(np.ones_like(w_u), 1)
        l_mask = u_mask.T

        w_p = torch.from_numpy(np.array(w_p))
        w_l = torch.from_numpy(np.array(w_l))
        w_s = torch.from_numpy(np.array(w_s))
        w_u = torch.from_numpy(np.array(w_u))

        self.register_buffer("w_p", w_p)
        self.register_buffer("u_mask", torch.from_numpy(u_mask))
        self.register_buffer("l_mask", torch.from_numpy(l_mask))
        self.register_buffer("s_sign", torch.sign(w_s))
        self.register_buffer("l_eye", torch.eye(l_mask.shape[0]))
        self.w_l = nn.Parameter(w_l)
        self.w_s = nn.Parameter(logabs(w_s))
        self.w_u = nn.Parameter(w_u)

    def forward(self, input):
        _, _, height, width = input.shape

        weight = self.calc_weight()

        out = F.conv2d(input, weight)
        logdet = height * width * torch.sum(self.w_s)

        return out, logdet

    def calc_weight(self):
        weight = (
            self.w_p
            @ (self.w_l * self.l_mask + self.l_eye)
            @ ((self.w_u * self.u_mask) + torch.diag(self.s_sign * torch.exp(self.w_s)))
        )

        return weight.unsqueeze(2).unsqueeze(3)

    def reverse(self, output):
        weight = self.calc_weight()

        return F.conv2d(output, weight.squeeze().inverse().unsqueeze(2).unsqueeze(3))


class ZeroConv2d(nn.Module):
    def __init__(self, in_channel, out_channel, padding=1):
        super().__init__()

        self.conv = nn.Conv2d(in_channel, out_channel, 3, padding=0)
        self.conv.weight.data.zero_()
        self.conv.bias.data.zero_()
        self.scale = nn.Parameter(torch.zeros(1, out_channel, 1, 1))

    def forward(self, input):
        out = F.pad(input, [1, 1, 1, 1], value=1)
        out = self.conv(out)
        out = out * torch.exp(self.scale * 3)

        return out


class AffineCoupling(nn.Module):
    def __init__(self, in_channel, filter_size=512, affine=True):
        super().__init__()

        self.affine = affine

        self.net = nn.Sequential(
            nn.Conv2d(in_channel // 2, filter_size, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(filter_size, filter_size, 1),
            nn.ReLU(inplace=True),
            ZeroConv2d(filter_size, in_channel if self.affine else in_channel // 2),
        )

        self.net[0].weight.data.normal_(0, 0.05)
        self.net[0].bias.data.zero_()

        self.net[2].weight.data.normal_(0, 0.05)
        self.net[2].bias.data.zero_()

    def forward(self, input):
        in_a, in_b = input.chunk(2, 1)

        if self.affine:
            log_s, t = self.net(in_a).chunk(2, 1)
            # s = torch.exp(log_s)
            s = F.sigmoid(log_s + 2)
            # out_a = s * in_a + t
            out_b = (in_b + t) * s

            logdet = torch.sum(torch.log(s).view(input.shape[0], -1), 1)
        else:
            net_out = self.net(in_a)
            out_b = in_b + net_out
            logdet = None

        return torch.cat([in_a, out_b], 1), logdet

    def reverse(self, output):
        out_a, out_b = output.chunk(2, 1)

        if self.affine:
            log_s, t = self.net(out_a).chunk(2, 1)
            # s = torch.exp(log_s)
            s = F.sigmoid(log_s + 2)
            # in_a = (out_a - t) / s
            in_b = out_b / s - t
        else:
            net_out = self.net(out_a)
            in_b = out_b - net_out

        return torch.cat([out_a, in_b], 1)


class Flow(nn.Module):
    def __init__(self, in_channel, affine=True, conv_lu=True):
        super().__init__()

        self.actnorm = ActNorm(in_channel)

        if conv_lu:
            self.invconv = InvConv2dLU(in_channel)
        else:
            self.invconv = InvConv2d(in_channel)

        self.coupling = AffineCoupling(in_channel, affine=affine)

    def forward(self, input):
        out, logdet = self.actnorm(input)
        out, det1 = self.invconv(out)
        out, det2 = self.coupling(out)

        logdet = logdet + det1
        if det2 is not None:
            logdet = logdet + det2

        return out, logdet

    def reverse(self, output):
        input = self.coupling.reverse(output)
        input = self.invconv.reverse(input)
        input = self.actnorm.reverse(input)

        return input


def gaussian_log_p(x, mean, log_sd):
    return -0.5 * log(2 * pi) - log_sd - 0.5 * (x - mean) ** 2 / torch.exp(2 * log_sd)


def gaussian_sample(eps, mean, log_sd):
    return mean + torch.exp(log_sd) * eps


class Block(nn.Module):
    def __init__(self, in_channel, n_flow, split=True, affine=True, conv_lu=True):
        super().__init__()

        squeeze_dim = in_channel * 4

        self.flows = nn.ModuleList()
        for _ in range(n_flow):
            self.flows.append(Flow(squeeze_dim, affine=affine, conv_lu=conv_lu))

        self.split = split

        if split:
            self.prior = ZeroConv2d(in_channel * 2, in_channel * 4)
        else:
            self.prior = ZeroConv2d(in_channel * 4, in_channel * 8)

    def forward(self, input):
        b_size, n_channel, height, width = input.shape
        squeezed = input.view(b_size, n_channel, height // 2, 2, width // 2, 2)
        squeezed = squeezed.permute(0, 1, 3, 5, 2, 4)
        out = squeezed.contiguous().view(b_size, n_channel * 4, height // 2, width // 2)

        logdet = 0

        for flow in self.flows:
            out, det = flow(out)
            logdet = logdet + det

        if self.split:
            out, z_new = out.chunk(2, 1)
            mean, log_sd = self.prior(out).chunk(2, 1)
            log_p = gaussian_log_p(z_new, mean, log_sd)
            log_p = log_p.view(b_size, -1).sum(1)
        else:
            zero = torch.zeros_like(out)
            mean, log_sd = self.prior(zero).chunk(2, 1)
            log_p = gaussian_log_p(out, mean, log_sd)
            log_p = log_p.view(b_size, -1).sum(1)
            z_new = out

        return out, logdet, log_p, z_new

    def reverse(self, output, eps=None, reconstruct=False):
        input = output

        if reconstruct:
            if self.split:
                input = torch.cat([output, eps], 1)
            else:
                input = eps
        else:
            if self.split:
                mean, log_sd = self.prior(input).chunk(2, 1)
                z = gaussian_sample(eps, mean, log_sd)
                input = torch.cat([output, z], 1)
            else:
                zero = torch.zeros_like(input)
                # zero = F.pad(zero, [1, 1, 1, 1], value=1)
                mean, log_sd = self.prior(zero).chunk(2, 1)
                z = gaussian_sample(eps, mean, log_sd)
                input = z

        for flow in self.flows[::-1]:
            input = flow.reverse(input)

        b_size, n_channel, height, width = input.shape

        unsqueezed = input.view(b_size, n_channel // 4, 2, 2, height, width)
        unsqueezed = unsqueezed.permute(0, 1, 4, 2, 5, 3)
        unsqueezed = unsqueezed.contiguous().view(
            b_size, n_channel // 4, height * 2, width * 2
        )

        return unsqueezed


class Glow(GenerativeModel):
    def __init__(self, params: argparse.Namespace, in_channel: int = 3):
        super().__init__(params)

        self.attr_vectors_dir = params.attr_vectors_dir

        self.img_size = params.image_size
        self.nc = in_channel
        self.n_flow = params.glow_n_flow
        self.n_block = params.glow_n_block
        self.affine = params.glow_affine
        self.conv_lu = not params.glow_no_lu

        self.z_shapes = self._calc_z_shapes(in_channel, self.img_size, self.n_block)

        self.blocks = nn.ModuleList()
        n_channel = in_channel
        for _ in range(self.n_block - 1):
            self.blocks.append(Block(n_channel, self.n_flow, affine=self.affine, conv_lu=self.conv_lu))
            n_channel *= 2
        self.blocks.append(Block(n_channel, self.n_flow, split=False, affine=self.affine))

    @staticmethod
    def _calc_z_shapes(n_channel, input_size, n_block):
        z_shapes = []
        for _ in range(n_block - 1):
            input_size //= 2
            n_channel *= 2
            z_shapes.append((n_channel, input_size, input_size))
        input_size //= 2
        z_shapes.append((n_channel * 4, input_size, input_size))
        return z_shapes

    def forward(self, input):
        log_p_sum = 0
        logdet = 0
        out = input
        z_outs = []

        for block in self.blocks:
            out, det, log_p, z_new = block(out)
            z_outs.append(z_new)
            logdet = logdet + det

            if log_p is not None:
                log_p_sum = log_p_sum + log_p

        return log_p_sum, logdet, z_outs

    def reverse(self, z_list, reconstruct=False):
        for i, block in enumerate(self.blocks[::-1]):
            if i == 0:
                input = block.reverse(z_list[-1], z_list[-1], reconstruct=reconstruct)
            else:
                input = block.reverse(input, z_list[-(i + 1)], reconstruct=reconstruct)

        return input

    def latent_dimension(self) -> int:
        d = sum(prod(z_shape) for z_shape in self.z_shapes)
        assert d == self.img_size * self.img_size * self.nc
        return d

    def latents(self, x: torch.Tensor) -> List[torch.Tensor]:
        _, _, z_outs = self(x)
        return z_outs

    def reconstruct_from_latents(self, z: List[torch.Tensor]) -> torch.Tensor:
        # Returned result in the [-0.5, 0.5] range.
        return self.reverse(z, reconstruct=True)

    def get_attribute_vectors(self, device, dtype) -> List[List[torch.Tensor]]:
        if self.attribute_vectors is not None:
            return self.attribute_vectors
        gen_model_file_or_dir = utils.get_path_to('saved_models') / self.params.gen_model_name
        attr_vectors_full_dir = utils.get_directory_of_path(gen_model_file_or_dir) / self.attr_vectors_dir
        assert attr_vectors_full_dir.is_dir()
        attr_names = self.params.perturb.split(',')
        self.attribute_vectors = []
        for attr_name in attr_names:
            if (attr_vectors_full_dir / f'{attr_name}.npz').is_file():
                self.attribute_vectors.append(
                    utils.load_attr_vector_npz_file(attr_vectors_full_dir / f'{attr_name}.npz', device, dtype)
                )
            else:
                npz_file_paths = list(sorted(attr_vectors_full_dir.glob(f'**/{attr_name}_*.npz')))

                def get_beg_end(npz_file_path: Path) -> Tuple[int, int]:
                    assert npz_file_path.name.startswith(f'{attr_name}_')
                    assert npz_file_path.suffix == '.npz'
                    beg_end_str = npz_file_path.name[len(f'{attr_name}_'):-len('.npz')]
                    underscore_index = beg_end_str.index('_')
                    return int(beg_end_str[:underscore_index]), int(beg_end_str[underscore_index + 1:])

                all_vals = set()
                for npz_file_path in npz_file_paths:
                    beg, end = get_beg_end(npz_file_path)
                    all_vals.add(beg)
                    all_vals.add(end)
                med = statistics.median(list(all_vals))

                for npz_file_path in npz_file_paths:
                    beg, end = get_beg_end(npz_file_path)
                    # if med - beg == end - med or end - beg >= med * 1.8:  # >= 13
                    if beg < med < end:
                        self.attribute_vectors.append(
                            utils.load_attr_vector_npz_file(npz_file_path, device, dtype)
                        )
        return self.attribute_vectors

    def unwrap_latents(self, wrapped_latents: torch.Tensor) -> List[torch.Tensor]:
        assert wrapped_latents.size(1) == self.latent_dimension()
        batch_size = wrapped_latents.size(0)
        idx = 0
        z_splits = []
        for shape in self.z_shapes:
            split_size = prod(shape)
            z_splits.append(
                wrapped_latents[:, idx : idx + split_size]
                    .view(batch_size, *shape)
            )
            idx += split_size
        return z_splits
