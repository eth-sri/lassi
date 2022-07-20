import argparse
from typing import List


def common() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Training/common arguments')

    parser.add_argument('--seed', type=int, default=42, help="Random seed")
    parser.add_argument('--use_cuda', type=bool_flag, default=True, help="Use GPU")
    parser.add_argument('--epochs', type=int, default=10, help="Number of training epochs")
    parser.add_argument('--batch_size', type=int, default=64, help="Batch size")
    parser.add_argument('--lr', type=float, default=0.0001, help="Learning rate")  # 1e-4.
    parser.add_argument('--parallel', type=bool_flag, default=False, help="Parallelize the training and evaluation")

    # Saving and logs:
    parser.add_argument('--save_artefacts', type=bool_flag, default=True,
                        help="Whether to save/log the results of the experiment")
    parser.add_argument('--save_period', type=int, default=0,
                        help="Save checkpoints on every save_period epochs. "
                             "If -1, save only the last one. If 0, do not save models")

    # Pipeline arguments:
    parser.add_argument('--train_encoder_batch_size', type=int, help="Batch size for training the encoder")
    parser.add_argument('--certification_batch_size', type=int, help="Batch size during certification")
    parser.add_argument('--train_classifier_batch_size', type=int, help="Batch size for training the classifier")

    return parser


def dataset() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Dataset arguments')

    parser.add_argument('--dataset', type=str, default='celeba',
                        choices=[
                            # CelebA:
                            'celeba',
                            'glow_celeba_64_latent_lmdb',
                            'glow_celeba_128_latent_lmdb',
                            # FairFace:
                            'fairface',
                            'glow_fairface_latent_lmdb',
                            # 3dshapes:
                            '3dshapes',
                            'glow_3dshapes_latent_lmdb',
                            'glow_3dshapes_latent_lmdb_correlated_orientation',
                        ],
                        help="Dataset name (only CelebA dataset supported so far)")
    parser.add_argument('--input_representation', type=str, default='original', choices=['original', 'latent'],
                        help="Are we providing original images, or latent representations in the generative model "
                             "latent space?")
    parser.add_argument('--image_size', type=int, default=64, help="Input image size")
    parser.add_argument('--n_bits', type=int, default=5,
                        help="Number of bits - used by Glow quantization preprocessing")
    parser.add_argument('--num_workers', type=int, default=4, help="Number of dataloader workers")

    # Classification task:
    parser.add_argument('--classify_attributes', type=argslist_str, default=[],
                        help="What attributes do we want to classify")

    return parser


def gen_model() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Generative model arguments')

    parser.add_argument('--gen_model_type', type=str, default='Glow', choices=['Glow'],
                        help="What generative model to be used")
    parser.add_argument('--gen_model_name', type=str,
                        help="Experiment/directory name to identify the trained generative model")
    parser.add_argument('--use_gen_model_reconstructions', type=bool_flag, default=False,
                        help="Use the generative model reconstructions, otherwise operate directly on its latent "
                             "representations")

    return parser


def glow() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Glow arguments')

    parser.add_argument('--glow_n_flow', type=int, default=32, help="Number of flows in each Glow block")
    parser.add_argument('--glow_n_block', type=int, default=4, help="Number of Glow blocks")
    # All Glow models pretrained by us use additive coupling (i.e. affine == False) and convolutional LU decomposition:
    parser.add_argument('--glow_no_lu', type=bool_flag, default=False,
                        help="Use plain convolution instead of LU decomposed version")
    parser.add_argument('--glow_affine', type=bool_flag, default=False,
                        help="Use affine coupling instead of additive")
    parser.add_argument('--attr_vectors_dir', type=str, default='attr_vectors_avg_diff',
                        help="Folder (inside the Glow saved_models directory) where the attribute vectors are located")

    return parser


def encoder() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Image encoder arguments')

    parser.add_argument('--encoder_type', type=str, default='linear', choices=['linear', 'resnet'],
                        help="(LASSI) encoder architecture")
    parser.add_argument('--encoder_use_bn', type=bool_flag, default=True,
                        help="Use batch normalization for the image encoders")
    parser.add_argument('--encoder_normalize_output', type=bool_flag, default=True,
                        help="Normalize the output of the encoder to have zero mean and standard deviation 1. This "
                             "is implemented by applying batch normalization to the last layer with the `affine` flag "
                             "set to False.")
    parser.add_argument('--resnet_encoder_pretrained', type=bool_flag, default=False,
                        help="Start with LASSI ResNet encoder pretrained on ImageNet")
    parser.add_argument('--encoder_hidden_layers', type=argslist_int, default=[],
                        help="Dimensions of the encoder hidden layers (considered only if the encoder type is linear)")

    # Classification loss weight:
    parser.add_argument('--cls_loss_weight', type=float, default=1.0, help="Classification loss weight")
    # (Auxiliary) downstream classifier architecture:
    parser.add_argument('--cls_layers', type=argslist_int, default=[],
                        help="Dimensions of the classifier hidden layers (separated by commas, no spaces in between)")

    return parser


def fair_encoder() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Fair (LASSI) encoder arguments')

    parser.add_argument('--fair_encoder_name', type=str,
                        help="Experiment/directory name to identify the trained fair LASSI encoder")
    parser.add_argument('--perturb', type=str, help="Specifies what semantic component/attribute(s) to perturb.")
    parser.add_argument('--perturb_epsilon', type=float,
                        help="Maximum magnitude of the perturbation. Vary the latent representation in the generative "
                             "model latent space by at most perturb_epsilon * ||attribute vector||_2")

    # Training:
    # 1) Data augmentation:
    parser.add_argument('--data_augmentation', type=bool_flag, default=False,
                        help="Augment the data with points from the similarity set "
                             "(or add Gaussian samples when performing standard classifier training)")
    # 2) Fairness constraints:
    parser.add_argument('--adv_loss_weight', type=float, default=0.0, help="Adversarial loss weight")
    parser.add_argument('--delta', type=float, default=0.0,
                        help="L2 distance enforced between latent representations of similar data points")
    parser.add_argument('--random_attack_num_samples', type=int, default=10,
                        help="Number of samples for the random attack")
    # 3) Reconstruction (transfer) loss:
    parser.add_argument('--recon_loss_weight', type=float, default=0.0, help="Reconstruction loss weight")
    parser.add_argument('--recon_decoder_type', type=str, default='linear', choices=['linear'],
                        help="Architecture of the reconstruction decoder")
    parser.add_argument('--recon_decoder_layers', type=argslist_int, default=[],
                        help="Dimensions of the reconstruction decoder hidden layers "
                             "(separated by commas, no spaces in between)")

    return parser


def fair_classifier() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Fair (LASSI) classifier arguments')

    parser.add_argument('--fair_classifier_name', type=str,
                        help="Experiment/directory name to identify the trained (with Gaussian noise augmentation) "
                             "fair LASSI classifier")
    parser.add_argument('--fair_classifier_data_augmentation', type=str, default='random_noise',
                        choices=['random_noise'],
                        help="Training method / data augmentation type of the fair classifier")
    parser.add_argument('--cls_sigma', type=float,
                        help="Standard deviation (sigma) of the Gaussian noise added to the LASSI classifier inputs")

    return parser


def certification() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Certification (smoothing) arguments')

    # LASSI encoder center smoothing parameters:
    parser.add_argument('--enc_sigma', type=float,
                        help="Standard deviation (sigma) of Gaussian noise added to "
                             "(gen model decoder + LASSI encoder) inputs in the direction of the attribute vector")
    parser.add_argument('--enc_alpha', type=float,
                        help="With probability at least 1 - alpha either abstain or return the prediction")
    parser.add_argument('--enc_n', type=int, default=1000000,
                        help="Number of samples required by the encoder for certification")
    parser.add_argument('--enc_n0', type=int, default=10000,
                        help="Smaller number of samples required by the encoder to estimate the center during "
                             "certification")

    # LASSI classifier Cohen smoothing parameters:
    parser.add_argument('--cls_alpha', type=float,
                        help="With probability at least 1 - alpha either abstain or return the prediction")
    parser.add_argument('--cls_n', type=int, default=100000,
                        help="Number of samples required by the classifier for prediction and certification")
    parser.add_argument('--cls_n0', type=int, default=2000,
                        help="Smaller number of samples required by the classifier to estimate the class during "
                             "certification")

    # What samples to certify:
    parser.add_argument('--sampling_batch_size', type=int, required=True,
                        help="Additional sampling batch size required by smoothing")
    parser.add_argument('--skip', type=int, default=1, help="Predict and ceritfy every skip samples via smoothing")
    parser.add_argument('--split', type=str, default='valid', choices=['valid', 'test'],
                        help="What split are we evaluating on?")

    # Certify options:
    parser.add_argument('--perform_endpoints_analysis', type=bool_flag, default=False,
                        help="Perform endpoints analysis")

    return parser


def bool_flag(arg: str) -> bool:
    if arg.lower() == 'true':
        return True
    elif arg.lower() == 'false':
        return False
    else:
        raise argparse.ArgumentTypeError('Invalid value for a boolean flag argument!')


def argslist_str(arg: str) -> List[str]:
    """Comma separated list of strings (no spaces in between)."""
    return [s.strip() for s in arg.split(',')]


def argslist_int(arg: str) -> List[int]:
    """Comma separated list of integers (no spaces in between)."""
    return [int(f) for f in argslist_str(arg)]


def argslist_float(arg: str) -> List[float]:
    """Comma separated list of floats (no spaces in between)."""
    return [float(f) for f in argslist_str(arg)]
