import numpy as np
from icudg.lib.hparams_registry import HparamSpec
from . import data
from ..multicenter.task import MulticenterICU


class PhysioNet2019(MulticenterICU):
    """Task setup for experiments using Physionet CinC 2019 challenge.

    Args:
        hparams (dict): a dictionary with all relevant hyperparameters
        args (dict): additional training arguments passed via the command line

    Environments: 
        'training_setA' : MIMIC IV (US)
        'training_setB' : Emory (US)
    """

    ENVIRONMENTS = ['training_setA', 'training_setB']
    
    num_classes = 2
    input_shape = None
    pad_to = 336
    
    HPARAM_SPEC = [
        # Data
        HparamSpec('val_env', 'train'),
        HparamSpec('test_env', 'training_setA'),

        # Training
        HparamSpec('lr', 1e-3, lambda r: float(np.exp(r.uniform(low=-10, high=-3)))),
        HparamSpec('batch_size', 128, lambda r: int(r.choice(a=[128, 256, 512]))),

        # Network
        HparamSpec('architecture', 'tcn'),
        HparamSpec('hidden_dims', 64, lambda r: int(r.choice(a=[32, 64, 128]))),
        HparamSpec('num_layers', 1, lambda r: int(r.randint(low=1, high=10))),
        HparamSpec('kernel_size', 4, lambda r: int(r.randint(low=2, high=6))),
        HparamSpec('heads', 4, lambda r: int(r.randint(low=1, high=3))),
        HparamSpec('dropout', 0.5, lambda r: float(r.choice(a=[0.3, 0.4, 0.5, 0.6, 0.7])))

    ]

    def __init__(self, hparams, args):
        self.args = args
        self.hparams = hparams
        self.envs = {e: data.PhysioNetEnvironment(e, self.pad_to) for e in self.ENVIRONMENTS}

        # Assign environments to train / val / test
        def _not(lst, excl):
            return [x for x in lst if x not in excl]

        self.TRAIN_ENVS = _not(self.ENVIRONMENTS, [hparams['val_env']] + [hparams['test_env']])
        if hparams['val_env'] == 'train':
            self.VAL_ENVS = self.TRAIN_ENVS
        else:
            self.VAL_ENVS = [hparams['val_env']]
        self.TEST_ENVS = [hparams['test_env']]

    def get_mask(self, batch):
        # batch: x, y, ...
        y = batch[1]
        return y != data.PAD_VALUE
