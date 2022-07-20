import argparse
from pathlib import Path
import random
import shutil
import sys

from loguru import logger
import matplotlib.pyplot as plt
import numpy as np
import torch


# Randomness, reproducibility and loading models:

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def common_experiment_setup(seed: int):
    """Basic (common) experiment setup: set the random seed and init the logger."""
    set_random_seed(seed)
    init_logger()


def load_model(model, checkpoint_file: Path, msg_model_name: str):
    logger.debug(f'Load {msg_model_name} from checkpoint file: {checkpoint_file}')
    model.load_state_dict(torch.load(checkpoint_file))
    model.eval()


# Show/save images:

def show(img):
    """
    Input is a tensor.

    NOTE: To save an image (or a set of images), use `torchvision.utils.save_image(...)` function, passing a tensor
        (single image or a batch) or a list of tensors and providing the path to the image to be saved (hint: file
        extension can be e.g. png, jpg, jpeg). This function can further receive the same arguments you would pass to
        `torchvision.utils.make_grid(...)`.
    """
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)), interpolation='nearest')
    plt.show()  # Blocking call.


# Device

def get_device(use_cuda: bool):
    return 'cuda' if use_cuda and torch.cuda.is_available() else 'cpu'


# Files and paths:

def get_root_path() -> Path:
    """Returns the path to fair-vision/."""
    return Path(__file__).resolve().parent.parent


def get_path_to(dir_name: str) -> Path:
    """Returns the path to fair-vision/data/."""
    assert dir_name in ['data', 'logs', 'saved_models', 'saved_images']
    return get_or_create_path(get_root_path() / dir_name)


def get_or_create_path(p: Path, empty_dir: bool = False) -> Path:
    """Create a folder if it does not already exist."""
    if not p.is_dir():
        p.mkdir(parents=True)
    else:
        if empty_dir:
            empty_folder(p)
    return p


def get_directory_of_path(p: Path):
    if p.is_dir():
        return p
    elif p.is_file():
        return p.parent
    else:
        raise ValueError(f'Provided path {p} neither directory nor file.')


def empty_folder(p: Path):
    """Delete the contents of a folder."""
    for child_path in p.iterdir():
        if child_path.is_file():
            child_path.unlink()
        elif child_path.is_dir():
            shutil.rmtree(child_path)


def get_checkpoint_file(models_dir: Path, prefix: str, load_checkpoint: int = -1):
    if load_checkpoint != -1:
        checkpoint_file = models_dir / f'{prefix}_{load_checkpoint}.pth'
        assert checkpoint_file.is_file()
        return checkpoint_file
    else:
        return get_last_checkpoint_file(models_dir, prefix)


def get_last_checkpoint_file(models_dir: Path, prefix: str):
    for suffix in ['last', 'best']:
        for ext in ['pth', 'pt']:
            if (models_dir / f'{prefix}_{suffix}.{ext}').is_file():
                checkpoint_file = models_dir / f'{prefix}_{suffix}.{ext}'
                assert checkpoint_file.is_file()
                return checkpoint_file
    max_checkpoint_epoch = -1
    checkpoint_file = None
    for ext in ['pth', 'pt']:
        for c in models_dir.glob(f'*.{ext}'):
            if c.name.startswith(prefix):
                checkpoint_epoch = int(c.stem.split('_')[-1])
                if checkpoint_epoch > max_checkpoint_epoch:
                    max_checkpoint_epoch = checkpoint_epoch
                    checkpoint_file = c
    assert checkpoint_file is not None
    assert checkpoint_file.is_file()
    return checkpoint_file


def load_attr_vector_npz_file(npz_file_path: Path, device, dtype):
    logger.debug('Load attribute vector from: {}'.format(npz_file_path))
    npz_file = np.load(npz_file_path)
    z_list = []
    for arr_name in npz_file.files:
        z_list.append(torch.from_numpy(npz_file[arr_name]).to(device=device, dtype=dtype))
    return z_list


# Logger:

LOGGER_FORMAT = '[{}][{}][{}] {}'.format(
        '<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green>',
        '<b><blue>{level: >7}</blue></b>',
        '<light-blue>{name}</light-blue>',
        '<cyan>{message}</cyan>'
    )


def init_logger():
    logger.remove()
    logger.add(sys.stderr, format=LOGGER_FORMAT)


def add_log_file(p: Path):
    return logger.add(p, format=LOGGER_FORMAT)


def spec_id(params: argparse.Namespace):
    attr_vectors_dir = params.attr_vectors_dir
    if attr_vectors_dir.startswith('released_'):
        attr_vectors_dir = attr_vectors_dir[len('released_'):]
    return 'perturb_{}_eps_{}_attr_vec_{}'.format(
        params.perturb, params.perturb_epsilon, attr_vectors_dir[len('attr_vectors_'):]
    )
