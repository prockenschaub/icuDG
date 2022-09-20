from functools import partial
import numpy as np
import pandas as pd

import torch
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
    num_classes = 2
    input_shape = None
    ES_PATIENCE = 7 # * checkpoint_freq steps
    
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
        self.d = data.MultiCenterDataset(
            hparams['outcome'], 
            self.TRAIN_PCT,
            self.VAL_PCT,
            args.seed
        )

        # Assign environments to train / val / test
        self.TRAIN_ENVS = _not(self.ENVIRONMENTS, [hparams['val_env']] + [hparams['test_env']])
        if hparams['val_env'] == 'train':
            self.VAL_ENVS = self.TRAIN_ENVS
        else:
            self.VAL_ENVS = [hparams['val_env']]
        self.TEST_ENVS = [hparams['test_env']]

        # Calculate case weights based on train fold of train envs
        train_data = pd.concat(self.get_datasets(self.TRAIN_ENVS, 'train'))
        prop_cases = np.mean(train_data.label)
        self.case_weight = (1 - prop_cases) / prop_cases

    def get_datasets(self, envs, dset):
        datasets = []
        for r in envs:
            ds = self.d[r][self.d[r]['fold'] == dset]
            datasets.append(ds)
        return datasets

    def get_torch_dataset(self, envs, dset):
        return ConcatDataset([data.SingleCenter(d) for d in self.get_datasets(envs, dset)])

    def get_loss_fn(self):
        return partial(bce_loss, pos_weight=self.case_weight)

    def get_mask(self, batch):
        _, y = batch
        return y != data.PAD_VALUE

    def get_featurizer(self, hparams):
        if hparams['architecture'] == "tcn":
            return featurizer.TCNet(
                self.d.num_inputs,
                hparams['hidden_dims'],
                hparams['num_layers'],
                hparams['kernel_size'],
                hparams['dropout']
            )
        elif hparams['architecture'] == "transformer":
            return featurizer.TransformerNet(
                self.d.num_inputs,
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
