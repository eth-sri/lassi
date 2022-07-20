import argparse
from typing import Optional

from loguru import logger
import torch
import torch.nn as nn

from models.generative_model import GenerativeModel, ParallelizableGenModelDecoder, ParallelizableGenModelEncoder
from models.glow import Glow
import utils


class GenModelFactory:

    @staticmethod
    def get(params: argparse.Namespace) -> GenerativeModel:
        if params.gen_model_type == 'Glow':
            logger.debug('Build Glow model with parameters: n_flow = {}, n_block = {}, no_lu = {}, affine = {}'.format(
                params.glow_n_flow, params.glow_n_block, params.glow_no_lu, params.glow_affine
            ))
            return Glow(params)
        else:
            raise ValueError(f'Generative models of type {params.gen_model_type} are not supported!')

    @staticmethod
    def get_and_load(params: argparse.Namespace, freeze: bool = True) -> GenerativeModel:
        assert params.gen_model_name is not None, 'Generative model name must be specified!'
        assert params.gen_model_type in ['Glow']

        file_or_dir = utils.get_path_to('saved_models') / params.gen_model_name
        assert file_or_dir.is_dir() or (file_or_dir.is_file() and file_or_dir.suffix in ['.pt', '.pth']), \
            'Cannot find generative model file nor its directory.'
        checkpoint_file = file_or_dir if file_or_dir.is_file() else utils.get_last_checkpoint_file(file_or_dir, 'model')

        model = GenModelFactory.get(params)
        utils.load_model(model, checkpoint_file, params.gen_model_type)
        if freeze:
            model.requires_grad_(False)
        return model


class GenModelWrapper:

    def __init__(self, params: argparse.Namespace, gen_model: Optional[GenerativeModel] = None):
        self.device = utils.get_device(params.use_cuda)
        if gen_model is None:
            self.gen_model = GenModelFactory.get_and_load(params, freeze=True).to(self.device)
        else:
            self.gen_model = gen_model
        self.parallel = (params.parallel & (torch.cuda.device_count() > 1))
        self.input_representation = params.input_representation
        self.use_gen_model_reconstructions = params.use_gen_model_reconstructions

        if self.parallel:
            if self.input_representation == 'original':
                self.gen_model_encoder = nn.DataParallel(
                    ParallelizableGenModelEncoder(self.gen_model)
                ).to(self.device)
            else:
                assert self.input_representation == 'latent'
                self.gen_model_encoder = None

            self.gen_model_decoder = nn.DataParallel(
                ParallelizableGenModelDecoder(self.gen_model)
            ).to(self.device)
        else:
            self.gen_model_encoder = None
            self.gen_model_decoder = None

    def get_latents_and_encoder_input(self, x):
        if self.parallel:
            if self.input_representation == 'original':
                z_gen_model_latents_wrapped = self.gen_model_encoder(x)
            else:
                assert self.input_representation == 'latent'
                z_gen_model_latents_wrapped = x

            if self.use_gen_model_reconstructions:
                enc_input = self.gen_model_decoder(z_gen_model_latents_wrapped)
            else:
                enc_input = z_gen_model_latents_wrapped.clone()
            z_gen_model_latents = self.gen_model.unwrap_latents(z_gen_model_latents_wrapped)
        else:
            if self.input_representation == 'original':
                z_gen_model_latents = self.gen_model.latents(x)
            else:
                assert self.input_representation == 'latent'
                z_gen_model_latents = self.gen_model.unwrap_latents(x)

            if self.use_gen_model_reconstructions:
                enc_input = self.gen_model.reconstruct_from_latents(z_gen_model_latents)
            else:
                enc_input = self.gen_model.wrap_latents(z_gen_model_latents)
        return z_gen_model_latents, enc_input

    def get_original_image(self, x):
        if self.input_representation == 'original':
            return x
        else:
            assert self.input_representation == 'latent'
            if self.parallel:
                assert self.gen_model_decoder is not None
                return self.gen_model_decoder(x)
            else:
                z_gen_model_latents = self.gen_model.unwrap_latents(x)
                return self.gen_model.reconstruct_from_latents(z_gen_model_latents)

    def get_wrapped_latents(self, x):
        # `x` represents a batch.
        if self.input_representation == 'latent':
            return x
        else:
            assert self.input_representation == 'original'
            if self.parallel:
                assert self.gen_model_encoder is not None
                return self.gen_model_encoder(x)
            else:
                latents = self.gen_model.latents(x)
                return self.gen_model.wrap_latents(latents)

    def compute_encoder_input(self, z_gen_model_latents_wrapped: torch.Tensor):
        if self.use_gen_model_reconstructions:
            if self.parallel:
                return self.gen_model_decoder(z_gen_model_latents_wrapped)
            else:
                z_gen_model_latents = self.gen_model.unwrap_latents(z_gen_model_latents_wrapped)
                return self.gen_model.reconstruct_from_latents(z_gen_model_latents)
        else:
            return z_gen_model_latents_wrapped
