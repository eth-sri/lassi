from enum import Enum
import json
from typing import List

from loguru import logger
import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score
import torch

from certification.center_smoothing import CenterSmoothing
from certification.classifier_smoothing import ClassifierSmoothing
from dataset.data_manager import DataManager


def get_x_and_y(x: torch.Tensor, y: torch.Tensor, device, data_manager: DataManager, skip: int,
                get_y: bool = True, get_full_y: bool = False):
    if get_full_y:
        full_y = y.clone()
    if get_y:
        y = data_manager.target_transform(y)

    if skip != 1:
        batch_indices = torch.arange(0, x.shape[0], step=skip, dtype=torch.long, device=x.device)
        x = x.index_select(dim=0, index=batch_indices)
        if get_full_y:
            full_y = full_y.index_select(dim=0, index=batch_indices)
        if get_y:
            y = y.index_select(dim=0, index=batch_indices)

    if get_y and get_full_y:
        return x.to(device), y.to(device), full_y.to(device)
    elif get_y and not get_full_y:
        return x.to(device), y.to(device)
    elif not get_y and get_full_y:
        return x.to(device), full_y.to(device)
    else:
        assert not get_y and not get_full_y
        return x.to(device)


class CertConsts(str, Enum):
    SAMPLE_INDEX = 'sample_index'
    GROUND_TRUTH_LABEL = 'ground_truth_label'

    CSM_RADIUS = 'csm_r'
    CSM_ERROR = 'csm_err'
    CSM_TIME = 'csm_t'

    SMOOTHED_CLS_PRED = 'smoothed_cls_p'
    CERTIFIED_CLS_RADIUS = 'certified_cls_r'
    CERTIFIED_FAIRNESS = 'certified_fair'
    COHEN_SMOOTHING_TIME = 'cls_sm_t'

    CERTIFIED_FAIRNESS_AND_ACCURACY = 'certified_fair_and_acc'

    EMPIRICALLY_FAIR = 'empirically_fair'
    CERTIFIED_FAIR_EMPIRICALLY_UNFAIR = 'certified_but_empirically_unfair'


class CertificationStatistics:

    def __init__(self):
        self.all_samples_certification_data: List[dict] = []

        self.csm_radii = []
        self.center_smoothing_errors = []

        self.ground_truth_labels = []
        self.smoothed_predictions = []

    @staticmethod
    def add_if_present(l: list, certification_data: dict, k: str):
        if k in certification_data:
            l.append(certification_data[k])

    def add(self, certification_data: dict):
        self.all_samples_certification_data.append(certification_data)

        if CertConsts.CSM_RADIUS in certification_data and certification_data[CertConsts.CSM_RADIUS] >= 0.0:
            self.csm_radii.append(certification_data[CertConsts.CSM_RADIUS])
        if CertConsts.CSM_ERROR in certification_data and certification_data[CertConsts.CSM_ERROR] >= 0.0:
            self.center_smoothing_errors.append(certification_data[CertConsts.CSM_ERROR])

        self.ground_truth_labels.append(certification_data[CertConsts.GROUND_TRUTH_LABEL])
        self.smoothed_predictions.append(
            certification_data.get(CertConsts.SMOOTHED_CLS_PRED, ClassifierSmoothing.ABSTAIN)
        )

    def total_count(self):
        return len(self.all_samples_certification_data)

    def get_acc(self):
        return 100.0 * accuracy_score(self.ground_truth_labels, self.smoothed_predictions)

    def get_balanced_acc(self):
        return 100.0 * balanced_accuracy_score(self.ground_truth_labels, self.smoothed_predictions)

    def get_cert_fair(self):
        cert_fair = [
            cert_data.get(CertConsts.CERTIFIED_FAIRNESS, False)
            for cert_data in self.all_samples_certification_data
        ]
        return 100.0 * np.mean(cert_fair)

    def get_cert_fair_and_acc(self):
        cert_fair_and_acc = [
            cert_data.get(CertConsts.CERTIFIED_FAIRNESS_AND_ACCURACY, False)
            for cert_data in self.all_samples_certification_data
        ]
        return 100.0 * np.mean(cert_fair_and_acc)

    def get_values(self, k: str):
        return [cert_data[k] for cert_data in self.all_samples_certification_data if k in cert_data]

    def get_mean_of_values(self, k: str, default_value_if_empty = 0.0):
        values = self.get_values(k)
        if values:
            return np.mean(values)
        else:
            return default_value_if_empty

    def num_of_csm_abstentions(self):
        return sum([
            (cert_data.get(CertConsts.CSM_RADIUS) == CenterSmoothing.ABSTAIN)
            for cert_data in self.all_samples_certification_data
        ])

    def num_of_classifier_abstentions(self):
        return sum([
            (cert_data.get(CertConsts.SMOOTHED_CLS_PRED) == ClassifierSmoothing.ABSTAIN)
            for cert_data in self.all_samples_certification_data
        ])

    def report(self):
        assert self.total_count() > 0
        logger.debug(f'[n = {self.total_count()}]:')

        csm_times_mean = self.get_mean_of_values(CertConsts.CSM_TIME)
        cohen_smoothing_times_mean = self.get_mean_of_values(CertConsts.COHEN_SMOOTHING_TIME)
        logger.debug('Times (sec) | csm - {:.2f}, csm + cls smoothing - {:.2f}'.format(
            csm_times_mean,
            csm_times_mean + cohen_smoothing_times_mean
        ))

        if self.csm_radii and self.center_smoothing_errors:
            logger.debug('Center smoothing average | radius = {:.5f}, error = {:.5f}'.format(
                np.mean(self.csm_radii), np.mean(self.center_smoothing_errors)
            ))
        logger.debug('Smoothing abstentions | center - {}, classifier - {} / out of {}'.format(
            self.num_of_csm_abstentions(), self.num_of_classifier_abstentions(), self.total_count()
        ))

        logger.debug(
            'Accuracy (%) | smoothed classification: normal = {:.2f}, balanced = {:.2f}'.format(
                self.get_acc(),
                self.get_balanced_acc()
            )
        )
        logger.debug(f'Certified fairness via smoothing (%) | {self.get_cert_fair():.2f}')
        logger.debug(f'Certifiably fair AND accurate (%) | {self.get_cert_fair_and_acc():.2f}')

        stdout_msg = {
            'n': self.total_count(),
            'csm_sec': csm_times_mean,
            'csm+cls_smoothing_sec': csm_times_mean + cohen_smoothing_times_mean,
            'acc': self.get_acc(),
            'balanced_acc': self.get_balanced_acc(),
            'cert_fair': self.get_cert_fair(),
            'fair_and_acc': self.get_cert_fair_and_acc()
        }

        if CertConsts.EMPIRICALLY_FAIR in self.all_samples_certification_data[0]:
            for cert_data in self.all_samples_certification_data:
                assert CertConsts.EMPIRICALLY_FAIR in cert_data
                assert CertConsts.CERTIFIED_FAIR_EMPIRICALLY_UNFAIR in cert_data
            logger.debug('Empirically fair (end points match and did not abstain) (%) | {:.2f}'.format(
                100.0 * self.get_mean_of_values(CertConsts.EMPIRICALLY_FAIR)
            ))
            logger.debug('Certifiably fair but empirically unfair (%) | {:.2f}'.format(
                100.0 * self.get_mean_of_values(CertConsts.CERTIFIED_FAIR_EMPIRICALLY_UNFAIR)
            ))

            stdout_msg['empirically_fair'] = 100.0 * self.get_mean_of_values(CertConsts.EMPIRICALLY_FAIR)
            stdout_msg['cert_fair_empirically_unfair'] = \
                100.0 * self.get_mean_of_values(CertConsts.CERTIFIED_FAIR_EMPIRICALLY_UNFAIR)

        print(json.dumps(stdout_msg))
