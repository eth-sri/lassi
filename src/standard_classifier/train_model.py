import argparse
import shutil

from loguru import logger
import torch
import torch.nn as nn
import torch.optim as optim

from abstract_trainer import AbstractTrainer
import args
from dataset.data_manager import DataManager
from metrics import ClassifierMetrics
from models.classifiers import MLPClassifier, OriginalImagesClassifier
from models.gen_model_factory import GenModelFactory, GenModelWrapper
from models.lassi_encoder import LASSIEncoderFactory
from sampler import GaussianNoiseAdder
import utils


def get_params() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a standard image classifier "
                    "(with or without Gaussian noise along the sensitive attribute vector)",
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


class StandardClassifierTrainer(AbstractTrainer):

    def __init__(self, params: argparse.Namespace, data_manager: DataManager):
        super(StandardClassifierTrainer, self).__init__(
            params, data_manager,
            maybe_save_models=True,
            empty_dir=True,
            log_file_name='train.log'
        )

        # Models:
        self.gen_model_wrapper = GenModelWrapper(params)
        if params.encoder_type in ['basic_cnn', 'resnet']:
            self.classifier = OriginalImagesClassifier(
                params,
                LASSIEncoderFactory.get_latent_dimension(params),
                num_classes=self.data_manager.num_classes()
            ).to(self.device)
        else:
            self.classifier = MLPClassifier(
                self.gen_model_wrapper.gen_model.latent_dimension(),
                params.encoder_hidden_layers,
                num_classes=self.data_manager.num_classes()
            ).to(self.device)
        self.ce_loss_fn = nn.CrossEntropyLoss(reduction='mean')
        self.opt = optim.Adam(self.classifier.parameters(), lr=params.lr)

        # Training with Gaussian noise (along the sensitive attribute vector).
        self.input_representation = params.input_representation
        self.data_augmentation = params.data_augmentation
        if self.data_augmentation:
            self.noise_adder = GaussianNoiseAdder(params.cls_sigma)

    @staticmethod
    def get_experiment_name(params: argparse.Namespace) -> str:
        if params.fair_classifier_name is not None:
            return params.fair_classifier_name
        experiment_name = 'std_cls/gen_model={}/'.format(params.gen_model_name)
        experiment_name += '_'.join(params.classify_attributes)
        enc_type = params.encoder_type
        if params.encoder_type == 'linear' and params.encoder_hidden_layers:
            enc_type += f'_{"_".join(map(str, params.encoder_hidden_layers))}'
        experiment_name += '_{}_seed_{}_epochs_{}_batch_size_{}_lr_{}_use_bn_{}'.format(
            enc_type, params.seed, params.epochs, params.batch_size, params.lr, params.encoder_use_bn
        )
        if params.encoder_type == 'resnet':
            experiment_name += f'_pretrained_{params.resnet_encoder_pretrained}'
        if params.data_augmentation:
            experiment_name += f'_perturb_{params.perturb}_sigma_{params.cls_sigma}'
        return experiment_name

    def _loop(self, epoch: int, subset: str, stats_only: bool):
        # Prepare data, metrics and loggers:
        metrics = ClassifierMetrics(subset)
        data_loader, pbar = self.get_data_loader_and_pbar(epoch, subset)

        for x, y in data_loader:
            y = self.data_manager.target_transform(y)
            x = x.to(self.device)
            y = y.to(self.device)

            if not self.data_augmentation and self.input_representation == 'original':
                enc_input = self.gen_model_wrapper.get_original_image(x)
            else:
                with torch.no_grad():
                    _, enc_input = self.gen_model_wrapper.get_latents_and_encoder_input(x)
                    z_gen_model_latents_wrapped = self.gen_model_wrapper.get_wrapped_latents(x)

            logits = self.classifier(enc_input)

            if self.data_augmentation:
                y_clean_pred = logits.argmax(dim=1).detach()
                with torch.no_grad():
                    z_gen_model_latents_wrapped = self.noise_adder.add_noise(
                        self.gen_model_wrapper.gen_model,
                        z_gen_model_latents_wrapped
                    )
                    enc_input = self.gen_model_wrapper.compute_encoder_input(z_gen_model_latents_wrapped)
                logits = self.classifier(enc_input)
            else:
                y_clean_pred = None

            y_pred = logits.argmax(dim=1)
            loss = self.ce_loss_fn(logits, y)

            if subset == 'train':
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()

            metrics.add(
                batch_loss=loss.item(), y_targets=y,
                y_pred=y_pred, y_clean_pred=y_clean_pred
            )

            pbar.update()

        # Wrap up:
        pbar.close()
        metrics.report(epoch)

        # The lower the score, the better.
        return -metrics.get_accuracies()[0]

    def run_loop(self, epoch: int = -1, subset: str = 'train', stats_only: bool = False):
        assert subset in ['train', 'valid', 'test']
        assert (subset == 'test' and epoch == -1) or (subset != 'test' and (epoch != -1 or stats_only))

        if subset == 'train' and not stats_only:
            self.classifier.train()
            return self._loop(epoch, subset, stats_only)
        else:
            self.classifier.eval()
            with torch.no_grad():
                return self._loop(epoch, subset, stats_only)

    def train(self):
        min_valid_score = float('inf')
        best_valid_epoch = -1

        self.start_timer()
        for epoch in range(self.epochs):
            self.run_loop(epoch, 'train')
            valid_score = self.run_loop(epoch, 'valid')
            model_was_saved = self.finish_epoch(epoch)
            if model_was_saved and min_valid_score > valid_score:
                min_valid_score = valid_score
                best_valid_epoch = epoch
        self.finish_training()

        if best_valid_epoch != -1:
            logger.debug(f'Best valid epoch: {best_valid_epoch}')
            logger.debug(f'Copy classifier_{best_valid_epoch + 1}.pth --> classifier_best.pth')

            with (self.logs_dir / 'best_valid_epoch').open('w') as f:
                f.write(f'Best valid epoch: {best_valid_epoch}\n')
                f.write(f'Copy classifier_{best_valid_epoch + 1}.pth --> classifier_best.pth\n')

            shutil.copyfile(
                self.saved_models_dir / f'classifier_{best_valid_epoch + 1}.pth',
                self.saved_models_dir / f'classifier_best.pth'
            )

    def save_models(self, checkpoint: str):
        torch.save(self.classifier.state_dict(), str(self.saved_models_dir / f'classifier_{checkpoint}.pth'))

    @staticmethod
    def load_models(params: argparse.Namespace, freeze: bool = True):
        # Training experiment artefacts:
        training_experiment_name = StandardClassifierTrainer.get_experiment_name(params)
        models_path = utils.get_path_to('saved_models') / training_experiment_name
        if models_path.is_file():
            assert models_path.suffix == '.pth'
            models_dir = models_path.parent
            checkpoint_suffix = models_path.stem.split('_')[-1]
            load_checkpoint = -1 if checkpoint_suffix in ['last', 'best'] else int(checkpoint_suffix)
        else:
            assert models_path.is_dir(), f'Directory not found: {models_path}'
            models_dir = models_path
            load_checkpoint = -1
        assert models_dir.is_dir(), \
            'Cannot find the saved_models sub-directory corresponding to the training experiment.'

        if params.encoder_type in ['basic_cnn', 'resnet']:
            classifier = OriginalImagesClassifier(
                params,
                LASSIEncoderFactory.get_latent_dimension(params),
                num_classes=DataManager.get_manager(params).num_classes()
            )
        else:
            classifier = MLPClassifier(
                GenModelFactory.get(params).latent_dimension(),
                params.encoder_hidden_layers,
                num_classes=DataManager.get_manager(params).num_classes()
            )

        classifier_checkpoint_file = utils.get_checkpoint_file(models_dir, 'classifier', load_checkpoint)
        utils.load_model(classifier, classifier_checkpoint_file, 'classifier')

        if freeze:
            classifier.requires_grad_(False)

        return classifier


def main(params: argparse.Namespace):
    assert params.gen_model_name is not None
    assert params.save_artefacts
    assert not params.data_augmentation or params.cls_sigma is not None

    data_manager = DataManager.get_manager(params)
    classifier_trainer = StandardClassifierTrainer(params, data_manager)

    # Training:
    logger.debug('Start training.')
    classifier_trainer.train()
    logger.debug('Training completed.')

    classifier_trainer.remove_log_file()
    return classifier_trainer.experiment_name


if __name__ == '__main__':
    params = get_params()
    utils.common_experiment_setup(params.seed)
    main(params)
