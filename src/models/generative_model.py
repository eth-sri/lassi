import argparse
from typing import List

import torch
import torch.nn as nn


class GenerativeModel(nn.Module):
    """A parent class for the generative models that will be used."""

    def __init__(self, params: argparse.Namespace):
        super(GenerativeModel, self).__init__()
        self.params = params
        self.attribute_vectors = None
        self.attribute_vectors_single_batch_wrapped = None

    def latent_dimension(self) -> int:
        """
        :return: The (total) dimensionality of the (combined and flattened) latent representations.
        """
        raise NotImplementedError('`latent_dimension` method not implemented!')

    def latents(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        For a given input batch of images, return their latent representations as a list. The concatenation of all
        tensors in the output list represents the full latent representation of the input images.

        :param x: Input batch of shape [batch_size, ...].
        :return: List of tensors, each of shape [batch_size, ...]. The shapes of the tensors in the list may not match.
        """
        raise NotImplementedError('`latents` method not implemented!')

    def reconstruct_from_latents(self, z: List[torch.Tensor]) -> torch.Tensor:
        """
        Provide output reconstructions from intermediate (batch of) latent representations. The output should be
        normalized, i.e. passed through sigmoid/tanh, etc. or any other necessary postprocessing such that would allow
        it to be passed as an input to a classifier further down in the pipeline.

        :param z: (Batch of) latent representations.
        :return: Postprocessed reconstruction of shape [batch_size, ...].
        """
        raise NotImplementedError('`reconstruct_from_latents` method not implemented!')

    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        """For a given input batch of images, return their (appropriately postprocessed) reconstructions.
        :param x: Input batch of shape [batch_size, ...].
        :return: Postprocessed reconstructions of shape [batch_size, ...].
        """
        z = self.latents(x)
        return self.reconstruct_from_latents(z)

    def wrap_latents(self, z: List[torch.Tensor]) -> torch.Tensor:
        """Wrap the latents from `List[torch.Tensor]` format to `torch.Tensor` essentially by flattening them.
        :param z: (Batch of) latent representations in the format described above, returned by
            :func:`GenerativeModel.latents`.
        :return: A single `torch.Tensor` latent representation constructed by flattening all intermediate latent
            representations in the input list and concatenating them (concatenating their rows, to be precise).
        """
        batch_size = z[0].size(0)
        z = [z_i.view(batch_size, -1) for z_i in z]
        return torch.cat(z, dim=1)

    def unwrap_latents(self, wrapped_latents: torch.Tensor) -> List[torch.Tensor]:
        """Unwrap (unflatten) the latents.
        :param wrapped_latents: The latents, represented as a `torch.Tensor`, with a shape
            [batch_size, total_latent_dimension], returned by e.g. calling :func:`GenerativeModel.wrap_latents`.
        :return: Return the latents in a format similar to the format of the :func:`GenerativeModel.latents` output,
            essentially reverting :func:`GenerativeModel.wrap_latents`.
        """
        raise NotImplementedError('`unwrap_latents` method not implemented!')

    def get_attribute_vectors(self, device, dtype) -> List[List[torch.Tensor]]:
        """
        Get the attribute vector (in the latent space) according to the perturbation mode (specified in the script
        arguments).
        :param device: To which device to load the attribute vector.
        :param dtype: Desired tensor type of the returned attribute vector.
        :return: The attribute vector in the latent space. The returned list must have the same number of elements as
            the result returned by :func:`GenerativeModel.latents` and the corresponding elements must have the same
            shapes (minus the batch size).
        """
        raise NotImplementedError('`get_attribute_vectors` method not implemented!')

    def get_attribute_vectors_single_batch_wrapped(self, device, dtype) -> List[torch.Tensor]:
        if self.attribute_vectors_single_batch_wrapped is not None:
            return self.attribute_vectors_single_batch_wrapped
        attr_vectors = self.get_attribute_vectors(device, dtype)
        attr_vectors_single_batch = [[z_i.unsqueeze(0) for z_i in attr_vector] for attr_vector in attr_vectors]
        self.attribute_vectors_single_batch_wrapped = [
            self.wrap_latents(attr_vector_single_batch) for attr_vector_single_batch in attr_vectors_single_batch
        ]
        return self.attribute_vectors_single_batch_wrapped

    def get_attribute_vectors_wrapped_repeated(self, device, dtype, repeats) -> List[torch.Tensor]:
        attr_vectors_wrapped = self.get_attribute_vectors_single_batch_wrapped(device, dtype)
        attr_vectors_wrapped_repeated = [
            torch.repeat_interleave(attr_vector_wrapped, repeats, dim=0) for attr_vector_wrapped in attr_vectors_wrapped
        ]
        return attr_vectors_wrapped_repeated


class ParallelizableGenModelEncoder(nn.Module):

    def __init__(self, gen_model: GenerativeModel):
        super(ParallelizableGenModelEncoder, self).__init__()
        self.gen_model = gen_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: Batch of input (images).
        :return: Their latents in wrapped form (the from of the output of :func:`GenerativeModel.wrap_latents`).
        """
        z = self.gen_model.latents(x)
        return self.gen_model.wrap_latents(z)


class ParallelizableGenModelDecoder(nn.Module):

    def __init__(self, gen_model: GenerativeModel):
        super(ParallelizableGenModelDecoder, self).__init__()
        self.gen_model = gen_model

    def forward(self, wrapped_latents: torch.Tensor) -> torch.Tensor:
        """
        :param wrapped_latents: Wrapped latents, as if returned by :func:`GenerativeModel.wrap_latents`.
        :return: Reconstruction images.
        """
        unwrapped_latents = self.gen_model.unwrap_latents(wrapped_latents)
        return self.gen_model.reconstruct_from_latents(unwrapped_latents)
