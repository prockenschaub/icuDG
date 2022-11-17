import abc
import torch

from icudg.algorithms.base import Algorithm
from icudg.lib.hparams_registry import HparamMixin

class Task(HparamMixin):
    
    @abc.abstractmethod
    def get_torch_dataset(self, envs, dset):
        pass

    @abc.abstractmethod
    def get_featurizer(self, hparams):
        pass

    @abc.abstractmethod
    def get_loss_fn(self):
        pass

    @abc.abstractmethod
    def get_extended_loss_fn(self):
        pass

    @abc.abstractmethod
    def get_mask(self, batch):
        pass

    @abc.abstractmethod
    def eval_metrics(
        self, 
        algorithm: Algorithm, 
        loader: torch.utils.data.DataLoader, 
        device: str, 
        **kwargs
    ):
        pass

    @abc.abstractmethod
    def save_task(self, file_path: str):
        pass
