import argparse
import shutil

from loguru import logger
import torch
import torch.nn as nn
import torch.optim as optim

from abstract_trainer import AbstractTrainer
import args
from dataset.data_manager import DataManager
from lassi_encoder.attack import RandomSamplingAttack
from metrics import ClassificationAndFairnessMetrics
from models.classifiers import lassi_classifier_factory
from models.common import LinearDecoder
from models.gen_model_factory import GenModelWrapper
from models.lassi_encoder import LASSIEncoderFactory
import utils


def get_params() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train or evaluate LASSI encoder",
        parents=[
            args.common(),
            args.dataset(),
            args.gen_model(),
            args.glow(),
            args.encoder(),
            args.fair_encoder()
        ],
        conflict_handler='resolve'
    )
    return parser.parse_args()


class FairEncoderExperiment(AbstractTrainer):

    def __init__(self, params: argparse.Namespace, data_manager: DataManager):
        super(FairEncoderExperiment, self).__init__(
            params, data_manager,
            maybe_save_models=True,
            empty_dir=False,
            log_file_name='train.log'
        )

        # Models:
        self.gen_model_wrapper = GenModelWrapper(params)

        # Create models and optimizer:
        self.encoder = LASSIEncoderFactory.get(params).to(self.device)
        self.classifier = lassi_classifier_factory(
            params,
            input_dim=self.encoder.latent_dimension(),
            num_classes=self.data_manager.num_classes()
        ).to(self.device)
        params_list = list(self.encoder.parameters()) + list(self.classifier.parameters())

        if params.recon_loss_weight > 0.0:
            self.recon_decoder_type = params.recon_decoder_type
            assert params.recon_decoder_type == 'linear'
            self.decoder = LinearDecoder(
                params,
                input_dimension=self.encoder.latent_dimension(),
                output_dimension=self.gen_model_wrapper.gen_model.latent_dimension()
            ).to(self.device)
            params_list += list(self.decoder.parameters())
        else:
            self.decoder = None
            self.decoder_loss_fn = None

        self.opt = optim.Adam(params_list, lr=params.lr)
        self.ce_loss_fn = nn.CrossEntropyLoss(reduction='mean')

        # Data augmentation, fairness constraints and arguments:
        self.perturb_epsilon = params.perturb_epsilon
        self.data_augmentation = params.data_augmentation
        if self.data_augmentation:
            self.data_augmentor = RandomSamplingAttack(params, self.gen_model_wrapper, self.encoder)
        self.adv_loss_weight = params.adv_loss_weight
        if self.adv_loss_weight > 0.0:
            assert params.perturb_epsilon is not None
            self.attack = RandomSamplingAttack(params, self.gen_model_wrapper, self.encoder)

        # Classification and reconstruction loss weights:
        self.cls_loss_weight = params.cls_loss_weight
        self.recon_loss_weight = params.recon_loss_weight

    @staticmethod
    def get_experiment_name(params: argparse.Namespace) -> str:
        if params.fair_encoder_name is not None:
            return params.fair_encoder_name
        target_attributes = '_'.join(params.classify_attributes)
        experiment_name = 'fair_encoder/gen_model={}_{}_dset_{}/'.format(
            params.gen_model_name,
            'recon' if params.use_gen_model_reconstructions else 'latent',
            params.dataset
        )
        if params.cls_loss_weight > 0.0:
            experiment_name += f'{target_attributes}_'
        experiment_name += 'cls_weight_{}_adv_weight_{}_recon_weight_{}'.format(
            params.cls_loss_weight, params.adv_loss_weight, params.recon_loss_weight
        )
        if params.adv_loss_weight > 0.0 or params.data_augmentation:
            experiment_name += '_' + utils.spec_id(params)
        experiment_name += '/'
        enc_type = params.encoder_type
        if params.encoder_type == 'linear' and params.encoder_hidden_layers:
            enc_type += f'_{"_".join(map(str, params.encoder_hidden_layers))}'
        experiment_name += '{}_seed_{}_epochs_{}_batch_size_{}_lr_{}_use_bn_{}_normalize_output_{}'.format(
            enc_type, params.seed, params.epochs, params.batch_size, params.lr, params.encoder_use_bn,
            params.encoder_normalize_output
        )
        if params.encoder_type == 'resnet':
            experiment_name += f'_pretrained_{params.resnet_encoder_pretrained}'
        if params.cls_layers:
            experiment_name += f'_cls_layers_{"_".join(map(str, params.cls_layers))}'
        if params.data_augmentation:
            experiment_name += f'_data_aug_num_samples_{params.random_attack_num_samples}'
        if params.adv_loss_weight > 0.0:
            if params.delta > 0.0:
                experiment_name += f'_delta_{params.delta}'
            experiment_name += '_uniform_attack_num_samples_{}'.format(params.random_attack_num_samples)
        if params.recon_loss_weight > 0.0 and params.recon_decoder_layers:
            experiment_name += f'_recon_decoder_layers_{"_".join(map(str, params.recon_decoder_layers))}'
        return experiment_name

    def _get_cls_loss_and_predictions(self, z_lassi, y):
        logits = self.classifier(z_lassi)
        classification_loss = self.ce_loss_fn(logits, y)
        predictions = logits.argmax(dim=1)
        return classification_loss, predictions

    @staticmethod
    def _get_l2_diffs_between_classes(z_lassi, y):
        assert z_lassi.dim() == 2 and y.dim() == 1 and z_lassi.size(0) == y.size(0)
        batch_size = z_lassi.size(0)

        z_lassi_1 = z_lassi.repeat(batch_size, 1)
        z_lassi_2 = z_lassi.repeat_interleave(batch_size, dim=0)

        y1 = y.repeat(batch_size)
        y2 = y.repeat_interleave(batch_size, dim=0)
        mask_eq = torch.eq(y1, y2)

        l2_diffs = torch.linalg.norm(z_lassi_1 - z_lassi_2, ord=2, dim=1)
        l2_different = torch.masked_select(l2_diffs, ~mask_eq)
        l2_same = torch.masked_select(l2_diffs, mask_eq)

        assert l2_different.shape[0] + l2_same.shape[0] == batch_size * batch_size

        return l2_different, l2_same

    def _loop(self, epoch: int, subset: str, stats_only: bool):
        # Prepare data, metrics and loggers:
        metrics = ClassificationAndFairnessMetrics(subset)
        data_loader, pbar = self.get_data_loader_and_pbar(epoch, subset)

        for x, y in data_loader:
            y = self.data_manager.target_transform(y)
            x = x.to(self.device)
            y = y.to(self.device)

            # --- Compute classification loss: ---
            z_gen_model_latents, enc_input = self.gen_model_wrapper.get_latents_and_encoder_input(x)
            z_lassi_train_mode = self.encoder(enc_input)

            if not self.data_augmentation:
                y_targets = y
                classification_loss, predictions = self._get_cls_loss_and_predictions(z_lassi_train_mode, y_targets)
            else:
                z_lassi_augmented, y_augmented = self.data_augmentor.augment_data(z_gen_model_latents, y)
                z_lassi_combined = torch.cat([z_lassi_train_mode, z_lassi_augmented])
                y_targets = torch.cat([y, y_augmented])
                classification_loss, predictions = self._get_cls_loss_and_predictions(z_lassi_combined, y_targets)

            # --- Compute class differences: ---
            l2_different, l2_same = self._get_l2_diffs_between_classes(z_lassi_train_mode, y)

            # --- Compute reconstruction loss: ---
            if self.recon_loss_weight > 0.0:
                assert self.decoder is not None
                assert self.recon_decoder_type == 'linear'
                glow_code = self.gen_model_wrapper.gen_model.wrap_latents(z_gen_model_latents)
                recon_code = self.decoder(z_lassi_train_mode)
                recon_loss = torch.linalg.norm(recon_code - glow_code, ord=2, dim=1).mean()
            else:
                recon_loss = 0.0

            # --- Compute adversarial loss: ---
            run_attack = (self.adv_loss_weight > 0.0)
            if run_attack:
                if subset == 'train' and not stats_only:
                    # Switch to eval mode for the adversarial attack.
                    self.encoder.eval()

                # Recompute z with the LASSI enc in eval mode (affects batch norm).
                z_lassi_eval_mode = self.encoder(enc_input)
                z_gen_model_latents_adv_wrapped = self.attack.get_adv_examples(z_gen_model_latents, z_lassi_eval_mode)

                enc_input_adv = self.gen_model_wrapper.compute_encoder_input(z_gen_model_latents_adv_wrapped)
                z_lassi_adv = self.encoder(enc_input_adv).detach()
                l2_diffs = torch.linalg.norm(z_lassi_eval_mode - z_lassi_adv, ord=2, dim=1)

                if subset == 'train' and not stats_only:
                    # Set back the training mode while actually computing the adversarial loss.
                    self.encoder.train()

                adv_loss = self.attack.calc_loss(z_gen_model_latents_adv_wrapped, z_lassi_train_mode).mean()
            else:
                adv_loss = 0.0

            total_loss = self.cls_loss_weight * classification_loss + \
                         self.adv_loss_weight * adv_loss + \
                         self.recon_loss_weight * recon_loss

            if subset == 'train' and not stats_only:
                self.opt.zero_grad()
                total_loss.backward()
                self.opt.step()

            # Add / compute the stats:
            metrics.add(
                num_samples_batch=x.size(0),
                total_loss_batch=total_loss.item(),
                classification_loss_batch=classification_loss.item(),
                y_targets=y_targets,
                y_pred=predictions,
                l2_different=l2_different,
                l2_same=l2_same,
                adv_loss_batch=adv_loss.item() if run_attack else 0.0,
                l2_diffs_batch=l2_diffs if run_attack else None,
                reconstruction_loss_batch=recon_loss.item() if self.recon_loss_weight > 0.0 else 0.0
            )
            pbar.set_postfix(
                tot_loss=total_loss.item(),
                cls_loss=classification_loss.item(),
                adv_loss=adv_loss.item() if run_attack else 0.0,
                l2_diff_adv=l2_diffs.mean().item() if run_attack else 0.0,
                recon_loss=recon_loss.item() if self.recon_loss_weight > 0.0 else 0.0
            )
            pbar.update()

        # Wrap up:
        pbar.close()
        metrics.report(epoch)

        if self.adv_loss_weight + self.recon_loss_weight > 0.0:
            return metrics.get_total_loss_score()
        else:
            # The lower the score, the better.
            return -metrics.get_acc_score()

    def run_loop(self, epoch: int = -1, subset: str = 'train', stats_only: bool = False):
        assert subset in ['train', 'valid', 'test']
        assert (subset == 'test' and epoch == -1) or (subset != 'test' and (epoch != -1 or stats_only))

        if subset == 'train' and not stats_only:
            self.encoder.train()
            self.classifier.train()
            if self.decoder is not None:
                self.decoder.train()
        else:
            self.encoder.eval()
            self.classifier.eval()
            if self.decoder is not None:
                self.decoder.eval()

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
            logger.debug(f'Copy lassi_encoder_{best_valid_epoch + 1}.pth --> lassi_encoder_best.pth')
            logger.debug(f'Copy classifier_{best_valid_epoch + 1}.pth --> classifier_best.pth')

            with (self.logs_dir / 'best_valid_epoch').open('w') as f:
                f.write(f'Best valid epoch: {best_valid_epoch}\n')
                f.write(f'Copy lassi_encoder_{best_valid_epoch + 1}.pth --> lassi_encoder_best.pth\n')
                f.write(f'Copy classifier_{best_valid_epoch + 1}.pth --> classifier_best.pth\n')

            shutil.copyfile(
                self.saved_models_dir / f'lassi_encoder_{best_valid_epoch + 1}.pth',
                self.saved_models_dir / f'lassi_encoder_best.pth'
            )
            shutil.copyfile(
                self.saved_models_dir / f'classifier_{best_valid_epoch + 1}.pth',
                self.saved_models_dir / f'classifier_best.pth'
            )

    def save_models(self, checkpoint: str):
        torch.save(self.encoder.state_dict(), str(self.saved_models_dir / f'lassi_encoder_{checkpoint}.pth'))
        torch.save(self.classifier.state_dict(), str(self.saved_models_dir / f'classifier_{checkpoint}.pth'))

    @staticmethod
    def load_models(params: argparse.Namespace, load_aux_classifier: bool = True, freeze: bool = True):
        # Training experiment artefacts:
        training_experiment_name = FairEncoderExperiment.get_experiment_name(params)
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

        # LASSI encoder and auxiliary classifier:
        lassi_encoder = LASSIEncoderFactory.get(params)
        lassi_encoder_checkpoint_file = utils.get_checkpoint_file(models_dir, 'lassi_encoder', load_checkpoint)
        utils.load_model(lassi_encoder, lassi_encoder_checkpoint_file, 'LASSI encoder')

        if freeze:
            lassi_encoder.requires_grad_(False)

        # Auxiliary classifier:
        if load_aux_classifier:
            aux_classifier = lassi_classifier_factory(
                params,
                input_dim=lassi_encoder.latent_dimension(),
                num_classes=DataManager.get_manager(params).num_classes()
            )
            aux_classifier_checkpoint_file = utils.get_checkpoint_file(models_dir, 'classifier', load_checkpoint)
            utils.load_model(aux_classifier, aux_classifier_checkpoint_file, 'auxiliary classifier')

            if freeze:
                aux_classifier.requires_grad_(False)

            return lassi_encoder, aux_classifier
        else:
            return lassi_encoder


def main(params: argparse.Namespace):
    assert params.gen_model_name is not None
    assert params.save_artefacts
    assert params.perturb is not None and params.perturb_epsilon is not None
    if params.use_gen_model_reconstructions:
        assert params.encoder_type == 'resnet'
    else:
        assert params.encoder_type == 'linear'
    # Loss weights:
    assert params.cls_loss_weight >= 0.0
    assert params.adv_loss_weight >= 0.0
    assert params.recon_loss_weight >= 0.0

    utils.set_random_seed(params.seed)

    data_manager = DataManager.get_manager(params)
    fair_encoder_experiment = FairEncoderExperiment(params, data_manager)

    if not (
        (fair_encoder_experiment.saved_models_dir / 'lassi_encoder_best.pth').is_file() and
        (fair_encoder_experiment.saved_models_dir / 'classifier_best.pth').is_file()
        ):
        # Training:
        logger.debug('Start training.')
        fair_encoder_experiment.train()
        logger.debug('Training completed.')
    else:
        logger.debug('Encoder already trained.')

    fair_encoder_experiment.remove_log_file()
    return fair_encoder_experiment.experiment_name


if __name__ == '__main__':
    params = get_params()
    utils.init_logger()
    main(params)
