# Original copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# 
# Based on code by Zhang et al., rearranged and refactored by 
# Patrick Rockenschaub. 

import abc

class ExperimentBase():
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

from clinicaldg.mnist.experiments import ColoredMNIST
from clinicaldg.eicu.experiments import (
    eICU, 
    eICUCorrLabel,
    eICUCorrNoise,
    eICUSubsampleUnobs,
    eICUSubsampleObs
)
from clinicaldg.cxr.experiments import (
    CXR,
    CXRBinary, 
    CXRSubsampleUnobs,
    CXRSubsampleObs
)
