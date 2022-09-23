from functools import partial
import numpy as np
import pandas as pd

import torch.nn.functional as F
from torch.utils.data import ConcatDataset

from clinicaldg.lib.misc import predict_on_set
from clinicaldg.lib.hparams_registry import HparamSpec
from clinicaldg.lib.metrics import roc_auc_score
from clinicaldg.experiments import base

from . import data, featurizer


def bce_loss(logits, y, mask, reduction='mean', pos_weight=None, **kwargs):
    logits = logits[..., -1]
    if pos_weight is not None:
        pos_weight = y.new_tensor(pos_weight)

    ce = F.binary_cross_entropy_with_logits(
        logits, 
        y, 
        reduction='none', 
        pos_weight=pos_weight,
        **kwargs
    )
    
    # Mask padded values when calculating the loss
    masked_ce = ce * mask

    # Aggregate as needed
    if reduction == 'mean':
        return masked_ce.sum() / mask.sum()
    elif reduction == 'sum':
        return masked_ce.sum()
    return masked_ce

def _not(lst, excl):
    return [x for x in lst if x not in excl]


class MultiCenter(base.Experiment):
    
    ENVIRONMENTS = ['mimic', 'eicu', 'hirid', 'aumc']
    TRAIN_PCT = 0.7
    VAL_PCT = 0.1
    MAX_STEPS = 2000
    N_WORKERS = 1
    CHECKPOINT_FREQ = 10
    ES_METRIC = 'loss'
    ES_MAXIMIZE = False
    ES_PATIENCE = 20 # * checkpoint_freq steps
    
    num_classes = 2
    input_shape = None
    
    HPARAM_SPEC = [
        # Data
        HparamSpec('outcome', 'sepsis'),
        HparamSpec('val_env', None),
        HparamSpec('test_env', 'mimic'),

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
        self.envs = {e: data.Environment(e, hparams['outcome']) for e in self.ENVIRONMENTS}

        # Assign environments to train / val / test
        self.TRAIN_ENVS = _not(self.ENVIRONMENTS, [hparams['val_env']] + [hparams['test_env']])
        if hparams['val_env'] == 'train':
            self.VAL_ENVS = self.TRAIN_ENVS
        else:
            self.VAL_ENVS = [hparams['val_env']]
        self.TEST_ENVS = [hparams['test_env']]

    def add_environment(self, name):
        self.envs[name] = data.Environment(name, self.hparams['outcome'])

    def setup(self, envs=None, use_weight=True):
        """Perform actual data loading and preprocessing"""
        if envs is None:
            envs = [e for e in self.envs.keys()]

        for name, obj in self.envs.items():
            if name in envs:
                obj.prepare(self.TRAIN_PCT, self.VAL_PCT, self.args.seed, self.args.debug)
        
        # Check that all have the same number of inputs
        input_dims = np.unique([e.num_inputs for e in self.envs.values() if e.loaded])
        if len(input_dims) > 1:
            raise ValueError(f'Different input dimensions in envs: {input_dims}')
        self.num_inputs = int(input_dims)

        # Calculate case weights based on train fold of train envs
        if use_weight:
            train_data = pd.concat([self.envs[e]['train'].data for e in self.TRAIN_ENVS])
            prop_cases = np.mean(train_data.label)
            self.case_weight = (1 - prop_cases) / prop_cases
        else:
            self.case_weight = None

    def get_torch_dataset(self, envs, dset):
        return ConcatDataset([self.envs[e][dset] for e in envs])

    def get_loss_fn(self):
        return partial(bce_loss, pos_weight=self.case_weight)

    def get_mask(self, batch):
        _, y = batch
        return y != data.PAD_VALUE

    def get_featurizer(self, hparams):
        if hparams['architecture'] == "tcn":
            return featurizer.TCNet(
                self.num_inputs,
                hparams['hidden_dims'],
                hparams['num_layers'],
                hparams['kernel_size'],
                hparams['dropout']
            )
        elif hparams['architecture'] == "transformer":
            return featurizer.TransformerNet(
                self.num_inputs,
                hparams['hidden_dims'],
                hparams['num_layers'],
                hparams['heads'],
                hparams['dropout']
            )
        return NotImplementedError(
            f"Architecture {hparams['architecture']} not available ",
            f"as a featurizer for the MultiCenter experiment"
        )

    def eval_metrics(self, algorithm, loader, device, **kwargs):
        logits, y, _ = predict_on_set(algorithm, loader, device)
        logits = logits[..., -1]

        # Obtain mask for predictions on padded values
        mask = y != data.PAD_VALUE
        
        # Get the "normal" masked logits for each time step
        logits = logits.view(-1)[mask.view(-1)].numpy()
        y = y.view(-1)[mask.view(-1)].long().numpy()

        return {'roc': roc_auc_score(y, logits)}
