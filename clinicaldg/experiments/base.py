import abc
import torch
from clinicaldg.lib.hparams_registry import HparamMixin

class Experiment(HparamMixin):
    
    @abc.abstractmethod
    def get_torch_dataset(self, envs, dset):
        pass

    @abc.abstractmethod
    def get_loss_fn(self):
        pass

    def get_mask(self, batch):
        _, y = batch
        return torch.ones_like(y)

    @abc.abstractmethod
    def get_featurizer(self, hparams):
        pass

    @abc.abstractmethod
    def predict_on_set(self, algorithm, loader, device):
        pass

    @abc.abstractmethod
    def eval_metrics(self, algorithm, loader, env_name, weights, device):
        pass