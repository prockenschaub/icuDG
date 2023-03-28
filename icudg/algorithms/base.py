# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# Modifications made by Patrick Rockenschaub
import abc
import numpy as np
import torch

from icudg.lib.hparams_registry import HparamSpec, HparamMixin


class Algorithm(torch.nn.Module, HparamMixin):
    """
    A subclass of Algorithm implements a domain generalization algorithm.
    Subclasses should implement the following:
    - update()
    - predict()
    """
    HPARAM_SPEC = [
        HparamSpec('lr', 1e-3, lambda r: 10**r.uniform(-7.0, -1)),
        HparamSpec('weight_decay', 0., lambda r: r.choice([0.] + (10.**np.arange(-7, -0)).tolist()))
    ]
    
    def __init__(self, task, num_domains, hparams):
        super(Algorithm, self).__init__()
        self.task = task
        self.loss_fn = task.get_loss_fn()
        self.num_domains = num_domains
        self.hparams = hparams

    @abc.abstractmethod
    def update(self, minibatches, device):
        """
        Perform one update step, given a list of (x, y) tuples for all
        environments.
        """
        pass

    @abc.abstractmethod
    def predict(self, x):
        pass

    @property
    def warmup(self):
        return False