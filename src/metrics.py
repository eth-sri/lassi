from abc import ABC, abstractmethod
from typing import Optional, Tuple

from loguru import logger
from sklearn.metrics import accuracy_score, balanced_accuracy_score
import torch
from torch.utils.tensorboard import SummaryWriter


class Metrics(ABC):

    def __init__(self, subset: str):
        assert subset in ['train', 'valid', 'test']
        self.subset = subset

    @staticmethod
    def get_logger_prefix(epoch: int, subset: str):
        assert subset in ['Train', 'Valid', 'Test']
        if subset in ['Train', 'Valid']:
            return 'Epoch {:2}, {} |'.format(epoch, subset)
        else:
            return 'Test |'

    @abstractmethod
    def report(self, epoch: int = -1, writer: Optional[SummaryWriter] = None, default_log_prefix_msg: str = ''):
        pass


class ClassifierMetrics(Metrics):

    def __init__(self, subset: str):
        super(ClassifierMetrics, self).__init__(subset)
        self.total_samples = 0
        self.loss_accumulator = 0.0
        self.y_targets = []
        self.y_predictions = []  # Might be predictions to noisy inputs in the case of training smoothed classifier.
        self.y_clean_predictions = []  # If training smoothed classifier, predictions to clean inputs.

    def add(self, batch_loss: float, y_targets, y_pred, y_clean_pred=None):
        self.total_samples += y_targets.numel()
        self.loss_accumulator += y_targets.numel() * batch_loss
        self.y_targets.append(y_targets.detach().cpu())
        self.y_predictions.append(y_pred.detach().cpu())
        if y_clean_pred is not None:
            self.y_clean_predictions.append(y_clean_pred.detach().cpu())

    def get_accuracies(self, y_preds=None) -> Tuple[float, float]:
        if y_preds is None:
            y_preds = self.y_predictions

        y_targets = torch.cat(self.y_targets).numpy()
        y_predictions = torch.cat(y_preds).numpy()
        assert y_targets.shape == y_predictions.shape
        assert y_targets.ndim == 1

        return (
            100 * accuracy_score(y_targets, y_predictions),
            100 * balanced_accuracy_score(y_targets, y_predictions)
        )

    def get_accuracies_str(self, y_preds=None) -> str:
        acc, balanced_acc = self.get_accuracies(y_preds)
        return f'Acc = {acc:.2f}, Balanced acc = {balanced_acc:.2f}'

    def get_mean_loss(self):
        return self.loss_accumulator / self.total_samples

    def report(self, epoch: int = -1, writer: Optional[SummaryWriter] = None, default_log_prefix_msg: str = ''):
        subset = self.subset.capitalize()
        logger_prefix = self.get_logger_prefix(epoch, subset)

        mean_loss = self.get_mean_loss()
        acc, balanced_acc = self.get_accuracies()
        log_message = '{}{} loss = {:.5f}, acc = {:.2f}, balanced acc = {:.2f}'.format(
            default_log_prefix_msg, logger_prefix, mean_loss, acc, balanced_acc
        )
        if self.y_clean_predictions:
            acc_clean, balanced_acc_clean = self.get_accuracies(self.y_clean_predictions)
            log_message += ', clean acc = {:.2f}, clean balanced acc = {:.2f}'.format(
                acc_clean, balanced_acc_clean
            )
        logger.debug(log_message)

        if writer is not None and epoch != -1:
            assert subset in ['Train', 'Valid']
            writer.add_scalar(f'{subset}/loss', mean_loss, global_step=epoch)
            writer.add_scalar(f'{subset}/acc', acc, global_step=epoch)
            writer.add_scalar(f'{subset}/balanced_acc', balanced_acc, global_step=epoch)
            if self.y_clean_predictions:
                writer.add_scalar(f'{subset}/clean_acc', acc_clean, global_step=epoch)
                writer.add_scalar(f'{subset}/clean_balanced_acc', balanced_acc_clean, global_step=epoch)


class ClassificationAndFairnessMetrics(Metrics):

    def __init__(self, subset: str):
        super(ClassificationAndFairnessMetrics, self).__init__(subset)
        self.classifier_metrics = ClassifierMetrics(subset)
        self.num_samples = 0
        self.total_loss = 0.0
        self.adv_loss = 0.0
        self.reconstruction_loss = 0.0
        self.l2_diffs = []
        self.l2_different_classes = []
        self.l2_same_classes = []
        self.qs = torch.tensor([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 0.999, 1.0])

    def add(self, num_samples_batch: int, total_loss_batch: float, classification_loss_batch: float, y_targets, y_pred,
            l2_different: torch.Tensor, l2_same: torch.Tensor, adv_loss_batch: float = 0.0, l2_diffs_batch = None,
            reconstruction_loss_batch: float = 0.0):
        self.classifier_metrics.add(classification_loss_batch, y_targets, y_pred)

        self.num_samples += num_samples_batch
        self.total_loss += total_loss_batch * num_samples_batch
        self.adv_loss += adv_loss_batch * num_samples_batch
        self.reconstruction_loss += reconstruction_loss_batch * num_samples_batch

        if l2_diffs_batch is not None:
            self.l2_diffs.append(l2_diffs_batch.detach().cpu())
        self.l2_different_classes.append(l2_different.detach().cpu())
        self.l2_same_classes.append(l2_same.detach().cpu())

    def report(self, epoch: int = -1, writer: Optional[SummaryWriter] = None, default_log_prefix_msg: str = ''):
        subset = self.subset.capitalize()
        logger_prefix = self.get_logger_prefix(epoch, subset)

        acc, balanced_acc = self.classifier_metrics.get_accuracies()

        log_message = '{} Losses: total = {:.5f}, cls = {:.5f}, adv = {:.5f}, recon = {:.5f}; '.format(
            logger_prefix,
            self.total_loss / self.num_samples,
            self.classifier_metrics.get_mean_loss(),
            self.adv_loss / self.num_samples,
            self.reconstruction_loss / self.num_samples
        )
        log_message += 'acc = {:.2f}, balanced acc = {:.2f}'.format(acc, balanced_acc)
        logger.debug(log_message)

        l2_same_classes = torch.cat(self.l2_same_classes)
        l2_different_classes = torch.cat(self.l2_different_classes)
        logger.debug('L2 diffs (same classes): mean = {:.5f}, min = {:.5f}, max = {:.5f}, med = {:.5f}'.format(
            l2_same_classes.mean().item(),
            l2_same_classes.min().item(),
            l2_same_classes.max().item(),
            l2_same_classes.median().item()
        ))
        logger.debug('L2 diffs (different classes): mean = {:.5f}, min = {:.5f}, max = {:.5f}, med = {:.5f}'.format(
            l2_different_classes.mean().item(),
            l2_different_classes.min().item(),
            l2_different_classes.max().item(),
            l2_different_classes.median().item()
        ))

        l2_diffs = None
        if self.l2_diffs:
            l2_diffs = torch.cat(self.l2_diffs)
            log_message = '{}: mean = {:.5f}, std = {:.5f}, min = {:.5f}, max = {:.5f}, med = {:.5f}\n'.format(
                'L2 diffs (similarity set)',
                l2_diffs.mean().item(),
                l2_diffs.std().item(),
                l2_diffs.min().item(),
                l2_diffs.max().item(),
                l2_diffs.median().item()
            )
            log_message += 'Quantiles / L2 diffs:\n{}\n{}'.format(
                [f'{q:.3f}' for q in self.qs.tolist()],
                [f'{v:.3f}' for v in l2_diffs.quantile(self.qs).tolist()]
            )
            logger.debug(log_message)

        if writer is not None and epoch != -1:
            assert subset in ['Train', 'Valid']
            writer.add_scalar(
                f'{subset}/total_loss',
                self.total_loss / self.num_samples,
                global_step=epoch
            )
            writer.add_scalar(
                f'{subset}/cls_loss',
                self.classifier_metrics.get_mean_loss(),
                global_step=epoch
            )
            writer.add_scalar(
                f'{subset}/adv_loss',
                self.adv_loss / self.num_samples,
                global_step=epoch
            )
            writer.add_scalar(
                f'{subset}/recon_loss',
                self.reconstruction_loss / self.num_samples,
                global_step=epoch
            )
            writer.add_scalar(f'{subset}/acc', acc, global_step=epoch)
            writer.add_scalar(f'{subset}/balanced_acc', balanced_acc, global_step=epoch)

            if l2_diffs is not None:
                writer.add_scalar(f'{subset}/l2_diffs_mean', l2_diffs.mean().item(), global_step=epoch)
                writer.add_scalar(f'{subset}/l2_diffs_std', l2_diffs.std().item(), global_step=epoch)
                writer.add_histogram(f'{subset}/l2_differences', l2_diffs, global_step=epoch)

            writer.add_histogram(f'{subset}/l2_diffs_same_classes', l2_same_classes, global_step=epoch)
            writer.add_histogram(f'{subset}/l2_diffs_different_classes', l2_different_classes, global_step=epoch)

    def get_acc_score(self) -> float:
        acc, _ = self.classifier_metrics.get_accuracies()
        return acc

    def get_total_loss_score(self):
        return self.total_loss / self.num_samples
