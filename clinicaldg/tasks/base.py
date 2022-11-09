import abc
from clinicaldg.lib.hparams_registry import HparamMixin

class Task(HparamMixin):
    
    @abc.abstractmethod
    def get_torch_dataset(self, envs, dset):
        pass

    @abc.abstractmethod
    def get_featurizer(self, hparams):
        pass
