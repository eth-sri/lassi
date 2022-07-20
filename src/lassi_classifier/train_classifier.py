import argparse

from loguru import logger
import torch
import torch.nn as nn
import torch.optim as optim

from abstract_trainer import AbstractTrainer
import args
from dataset.data_manager import DataManager
from lassi_encoder.train_encoder import FairEncoderExperiment
from metrics import ClassifierMetrics
from models.classifiers import lassi_classifier_factory
from models.gen_model_factory import GenModelWrapper
from models.lassi_encoder import LASSIEncoderFactory
import utils


def get_params() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train LASSI classifier (augmented with Gaussian noise)",
        parents=[
            args.common(),
            args.dataset(),
            args.gen_model(),
            args.glow(),
            args.encoder(),
            args.fair_encoder(),
            args.fair_classifier()
        ],
        conflict_handler='resolve'
    )
    return parser.parse_args()


class FairClassifierTrainer(AbstractTrainer):
    """
    Train the LASSI classifier with added Gaussian noise to its input
    (i.e. Gaussian noise added in the LASSI encoder latent space).
    """

    def __init__(self, params: argparse.Namespace, data_manager: DataManager):
        super(FairClassifierTrainer, self).__init__(params, data_manager, maybe_save_models=True, empty_dir=False)

        self.input_representation = params.input_representation

        # Models:
        self.gen_model_wrapper = GenModelWrapper(params)
        self.lassi_encoder = FairEncoderExperiment.load_models(
            params, load_aux_classifier=False, freeze=True
        ).to(self.device)

        # Create the fair (LASSI) classifier that we will be training:
        self.lassi_classifier = lassi_classifier_factory(
            params,
            input_dim=self.lassi_encoder.latent_dimension(),
            num_classes=self.data_manager.num_classes()
        ).to(self.device)

        # Loss function and optimizer:
        self.ce_loss_fn = nn.CrossEntropyLoss(reduction='mean')
        self.opt = optim.Adam(self.lassi_classifier.parameters(), lr=params.lr)

        # Training / data augmentation parameters:
        self.data_augmentation = params.fair_classifier_data_augmentation
        # - Standard deviation of the noise. The noise is added in the LASSI encoder latent space!
        self.cls_sigma = params.cls_sigma

        if self.data_augmentation == 'random_noise':
            assert self.cls_sigma is not None
        else:
            raise ValueError(f'Data augmentation {self.data_augmentation} not supported!')

    @staticmethod
    def get_experiment_name(params: argparse.Namespace) -> str:
        if params.fair_classifier_name is not None:
            return f'{params.fair_encoder_name}/{params.fair_classifier_name}'
        target_attributes = '_'.join(params.classify_attributes)
        experiment_name = '{}/fair_classifier_{}_epochs_{}_batch_size_{}_lr_{}_cls_sigma_{}'.format(
            params.fair_encoder_name, target_attributes, params.epochs, params.batch_size, params.lr, params.cls_sigma
        )
        if params.cls_layers:
            experiment_name += f'_cls_layers_{"_".join(map(str, params.cls_layers))}'
        return experiment_name

    def _loop(self, epoch: int, subset: str):
        # Data, metrics and loggers:
        classifier_metrics = ClassifierMetrics(subset)
        data_loader, pbar = self.get_data_loader_and_pbar(epoch, subset)

        for x, y in data_loader:
            y = self.data_manager.target_transform(y)
            x = x.to(self.device)
            y = y.to(self.device)

            _, enc_input = self.gen_model_wrapper.get_latents_and_encoder_input(x)
            z = self.lassi_encoder(enc_input)

            # Clean inputs:
            logits_clean = self.lassi_classifier(z).detach()

            # Corrupted inputs:
            if self.data_augmentation == 'random_noise':
                noise = torch.randn_like(z, device=z.device) * self.cls_sigma
                z = z + noise
            else:
                raise ValueError(f'Data augmentation `{self.data_augmentation}` not supported yet')

            logits_corrupted = self.lassi_classifier(z)
            loss = self.ce_loss_fn(logits_corrupted, y)

            if subset == 'train':
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()

            predictions_clean = logits_clean.argmax(dim=1)
            predictions_corrupted = logits_corrupted.argmax(dim=1)
            classifier_metrics.add(
                batch_loss=loss.item(), y_targets=y,
                y_pred=predictions_corrupted, y_clean_pred=predictions_clean
            )

            pbar.update()

        # Wrap up:
        pbar.close()
        classifier_metrics.report(epoch)

    def run_loop(self, epoch: int = -1, subset: str = 'train', **kwargs):
        assert subset in ['train', 'valid', 'test']
        assert (subset == 'test' and epoch == -1) or (subset != 'test' and epoch != -1)

        if subset == 'train':
            self.lassi_classifier.train()
            self._loop(epoch, subset)
        else:
            self.lassi_classifier.eval()
            with torch.no_grad():
                self._loop(epoch, subset)

    def save_models(self, checkpoint: str):
        torch.save(
            self.lassi_classifier.state_dict(),
            str(self.saved_models_dir / f'lassi_classifier_{checkpoint}.pth')
        )

    @staticmethod
    def load_model(params: argparse.Namespace, freeze: bool = True) -> nn.Module:
        # Recreate the experiment name - critical to have the required command line arguments passed again.
        training_experiment_name = FairClassifierTrainer.get_experiment_name(params)
        model_dir = utils.get_path_to('saved_models') / training_experiment_name
        logger.debug('Load the model obtained during the following fair classifier training experiment:')
        logger.debug(training_experiment_name)
        assert model_dir.is_dir(), \
            'Cannot find the saved_models sub-directory corresponding to the training experiment.'

        # Re-build LASSI classifier.
        lassi_classifier = lassi_classifier_factory(
            params,
            LASSIEncoderFactory.get_latent_dimension(params),
            num_classes=DataManager.get_manager(params).num_classes()
        )

        # Load checkpoint:
        # Only support loading the last checkpoint for now.
        lassi_classifier_checkpoint_file = utils.get_checkpoint_file(model_dir, 'lassi_classifier', -1)
        utils.load_model(lassi_classifier, lassi_classifier_checkpoint_file, 'LASSI classifier')
        if freeze:
            lassi_classifier.requires_grad_(False)

        return lassi_classifier


def main(params: argparse.Namespace):
    assert params.gen_model_name is not None and params.fair_encoder_name is not None
    assert params.save_artefacts

    utils.set_random_seed(params.seed)

    dset_manager = DataManager.get_manager(params)
    fair_classifier_trainer = FairClassifierTrainer(params, dset_manager)

    if not (fair_classifier_trainer.saved_models_dir / f'lassi_classifier_{params.epochs}.pth').is_file():
        logger.debug('Start training.')
        fair_classifier_trainer.train()
        logger.debug('Training completed.')
    else:
        logger.debug('Classifier already trained.')

    fair_classifier_trainer.remove_log_file()
    if params.save_artefacts:
        return fair_classifier_trainer.experiment_name


if __name__ == '__main__':
    params = get_params()
    utils.init_logger()
    main(params)
