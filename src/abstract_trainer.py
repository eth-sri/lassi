import sys
from abc import ABC, abstractmethod
import argparse
import datetime
import json
import time
from typing import Optional

from loguru import logger
from tqdm import tqdm

from dataset.data_manager import DataManager
import utils


class AbstractTrainer(ABC):

    def __init__(self, params: argparse.Namespace, data_manager: DataManager, maybe_save_models: bool = False,
                 empty_dir: bool = True, log_file_name: Optional[str] = 'experiment.log'):
        self.device = utils.get_device(params.use_cuda)
        self.data_manager = data_manager
        self.epochs = params.epochs

        # Experiment artefacts:
        self.save_artefacts = params.save_artefacts
        self.save_period = params.save_period
        self.create_log_file = False
        if self.save_artefacts:
            self.experiment_name = self.get_experiment_name(params)
            self.logs_dir = utils.get_or_create_path(
                utils.get_path_to('logs') / self.experiment_name,
                empty_dir=empty_dir
            )
            if empty_dir:
                with (self.logs_dir / 'params.json').open('w') as f:
                    json.dump({
                        'args': vars(params),
                        'cmd': 'python ' + ' '.join(sys.argv)
                    }, f, sort_keys=True, indent=4)
            if log_file_name is not None:
                log_file = self.logs_dir / log_file_name
                self.log_file_id = utils.add_log_file(log_file)
                self.create_log_file = True
            logger.debug(f'Running experiment: {self.experiment_name}')
            # if empty_logs_dir:
            #     self.writer = SummaryWriter(str(self.logs_dir), flush_secs=60)

            # Further experiment artefacts - model saving:
            if maybe_save_models:
                if self.save_period != 0:
                    self.saved_models_dir = utils.get_or_create_path(
                        utils.get_path_to('saved_models') / self.experiment_name, empty_dir=empty_dir
                    )
                    if empty_dir:
                        with (self.saved_models_dir / 'hparams.json').open('w') as f:
                            json.dump(vars(params), f, sort_keys=False, indent=4)

        # Timing (set at the start of the training, if needed):
        self.start_time = None

    @staticmethod
    @abstractmethod
    def get_experiment_name(params: argparse.Namespace) -> str:
        raise NotImplementedError('`get_experiment_name` method not implemented.')

    def start_timer(self):
        self.start_time = time.time()

    def get_data_loader_and_pbar(self, epoch: int, subset: str):
        assert subset in ['train', 'valid', 'test']

        data_loader = self.data_manager.get_dataloader(subset)
        pbar = tqdm(total=len(data_loader), leave=False, ncols=175)
        if self.start_time is None:
            self.start_timer()
        time_delta = int(time.time() - self.start_time)
        if epoch != -1:
            pbar.set_description('[{}, {}_epoch = {:3}]'.format(
                str(datetime.timedelta(seconds=time_delta)), subset, epoch))
        else:
            pbar.set_description('[{}, test]'.format(str(datetime.timedelta(seconds=time_delta))))

        return data_loader, pbar

    @abstractmethod
    def run_loop(self, epoch: int, subset: str, **kwargs):
        raise NotImplementedError('`run_loop` method not implemented!')

    def train(self):
        self.start_timer()
        for epoch in range(self.epochs):
            self.run_loop(epoch, 'train')
            self.run_loop(epoch, 'valid')
            self.finish_epoch(epoch)
        self.finish_training()

    def finish_epoch(self, epoch: int):
        # Return if model is saved.
        if self.save_artefacts and self.save_period > 0 and (epoch + 1) % self.save_period == 0:
            self.save_models(str(epoch + 1))
            return True
        return False

    def finish_training(self):
        if self.save_artefacts:
            if hasattr(self, 'writer'):
                self.writer.close()
            if self.save_period == -1:
                self.save_models('last')

    @abstractmethod
    def save_models(self, checkpoint: str):
        raise NotImplementedError('`save_models` method not implemented!')

    def remove_log_file(self):
        if self.create_log_file:
            logger.remove(self.log_file_id)
