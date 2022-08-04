# Original copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# 
# Based on code by Zhang et al., rearranged and refactored by 
# Patrick Rockenschaub. 

import abc

from clinicaldg.lib.hparams_registry import HparamMixin

class ExperimentBase(HparamMixin):
    
    @abc.abstractmethod
    def get_torch_dataset(self, envs, dset):
        pass

    @abc.abstractmethod
    def get_featurizer(self, hparams):
        pass

    @abc.abstractmethod
    def predict_on_set(self, algorithm, loader, device):
        pass

    @abc.abstractmethod
    def eval_metrics(self, algorithm, loader, env_name, weights, device):
        pass

from .cxr.experiments import (
    CXR,
    CXRBinary, 
    CXRSubsampleUnobs,
    CXRSubsampleObs
)
from .eicu.experiments import (
    eICU, 
    eICUCorrLabel,
    eICUCorrNoise,
    eICUSubsampleUnobs,
    eICUSubsampleObs
)
from .mnist.experiments import ColoredMNIST
from .multicenter.experiments import MultiCenterMimic