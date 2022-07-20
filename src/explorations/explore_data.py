import argparse

import torch
import torchvision

from dataset.data_manager import DataManager
from models.gen_model_factory import GenModelFactory

import utils


def glow_reconstructions(dataset: str, attr_vectors_dir: str, perturb, perturb_epsilon):
    if 'fairface' in dataset:
        gen_model_name = 'glow_fairface'
    elif 'celeba' in dataset:
        gen_model_name = 'glow_celeba_64'
    else:
        raise ValueError

    params = argparse.Namespace(
        use_cuda=True, batch_size=1, skip=1,
        dataset=dataset, image_size=64, n_bits=5, num_workers=4,
        classify_attributes=[],
        gen_model_type='Glow', gen_model_name=gen_model_name,
        glow_n_flow=32, glow_n_block=4, glow_no_lu=False, glow_affine=False, attr_vectors_dir=attr_vectors_dir,
        perturb=perturb, perturb_epsilon=perturb_epsilon
    )

    device = utils.get_device(params.use_cuda)

    data_manager = DataManager.get_manager(params)
    glow = GenModelFactory.get_and_load(params).to(device)
    attr_vector = None

    def attribute_vector_info(z_delta):
        print('Attribute vector info:')
        z_flattened = []
        print('Shapes:')
        for z_i in z_delta:
            print(z_i.shape)
            z_flattened.append(z_i.view(-1))
        print('Length:')
        z_flattened = torch.cat(z_flattened)
        print(z_flattened.shape)
        print(torch.linalg.norm(z_flattened))
        print(torch.linalg.norm(z_flattened, ord=2))
        print(z_flattened)

    cnt = 0
    for x, _ in data_manager.get_dataloader('train', shuffle=False):
        x = x.to(device)
        if params.dataset in ['fairface', 'celeba']:
            x = glow.wrap_latents(glow.latents(x))

        if params.skip != 1:
            batch_indices = torch.arange(0, x.size(0), step=params.skip, dtype=torch.long, device=device)
            x = x.index_select(dim=0, index=batch_indices)

        if attr_vector is None:
            attr_vector = glow.get_attribute_vectors(device, x.dtype)
            assert len(attr_vector) == 1
            attr_vector = attr_vector[0]
            attribute_vector_info(attr_vector)
            attr_vector = [z_i.unsqueeze(0) for z_i in attr_vector]
            attr_vector = glow.wrap_latents(attr_vector)
            assert attr_vector.shape == x.shape

        coeffs = [-1.0, -0.5, 0.0, 0.5, 1.0]
        x_recons = []
        for l in coeffs:
            x_recon = glow.reconstruct_from_latents(glow.unwrap_latents(x + l * perturb_epsilon * attr_vector)).cpu()
            x_recons.append(x_recon)

        reconstructions = torch.stack(x_recons, dim=1).view(-1, 3, params.image_size, params.image_size)
        grid = torchvision.utils.make_grid(reconstructions, normalize=True, range=(-0.5, 0.5), nrow=len(coeffs))
        utils.show(grid)

        cnt += 1
        inp = input(f'Continue to {cnt} [y/n]?')
        if inp == 'n':
            break


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Data exploration (and debugging)")
    parser.add_argument('--dataset', type=str, help="Dataset name")
    parser.add_argument('--glow_reconstructions', action='store_true', default=False,
                        help="Show selected Glow reconstructions")
    parser.add_argument('--attr_vectors_dir', type=str, default='attr_vectors_avg_diff',
                        help="Attribute vector directory")
    parser.add_argument('--perturb', help="What attribute to perturb")
    parser.add_argument('--perturb_epsilon', type=float, help="By how much can we perturb?")
    params = parser.parse_args()

    utils.init_logger()

    if params.glow_reconstructions:
        glow_reconstructions(params.dataset, params.attr_vectors_dir, params.perturb, params.perturb_epsilon)
