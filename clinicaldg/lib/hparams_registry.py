# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import numpy as np
from . import misc


class HparamSpec():
    def __init__(self, name, default_val, random_val_fn = None):
        self.name = name
        self.default_val = default_val
        self.random_val_fn = random_val_fn
    
    @property
    def default(self):
        return self.default_val

    def random(self, random_seed = None):
        random_state = np.random.RandomState(
            misc.seed_hash(random_seed, self.name)
        )
        if self.random_val_fn is not None:
            return self.random_val_fn(random_state)
        else:
            return self.default
        

class HparamRegistry():
    def __init__(self) -> None:
        self.hparams = {}

    def register(self, hparams):
        for h in hparams:
            self.hparams[h.name] = h

    def get_defaults(self):
        return {k: v.default for k, v in self.hparams.items()}

    def get_random_instance(self, random_seed):
        return {k: v.random(random_seed) for k, v in self.hparams.items()}


class HparamMixin():
    """A class for all objects that define hyperparameters (e.g., algorithms
    or experiments)"""
    
    HPARAM_SPEC = []
