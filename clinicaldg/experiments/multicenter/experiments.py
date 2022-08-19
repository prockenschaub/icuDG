import numpy as np

import torch.nn.functional as F
from torch.utils.data import ConcatDataset

from clinicaldg.lib.misc import predict_on_set
from clinicaldg.lib.hparams_registry import HparamSpec
from clinicaldg.lib.metrics import roc_auc_score
from clinicaldg.experiments import base

from . import data, featurizer


def bce_loss(logits, y, mask, reduction='mean', **kwargs):
    logits = logits[..., -1]
    ce = F.binary_cross_entropy_with_logits(
        logits, 
        y, 
        reduction='none', 
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


class MultiCenterBase(base.Experiment):
    
    ENVIRONMENTS = ['mimic', 'eicu', 'hirid', 'aumc']
    TRAIN_PCT = 0.7
    VAL_PCT = 0.1
    MAX_STEPS = 2000
    N_WORKERS = 1
    CHECKPOINT_FREQ = 10
    ES_METRIC = 'roc'
    num_classes = 2
    input_shape = None
    ES_PATIENCE = 7 # * checkpoint_freq steps
    
    HPARAM_SPEC = [
        # Data
        HparamSpec('mc_outcome', 'sepsis'),
        HparamSpec('mc_test_env', 'aumc'),

        # Training
        HparamSpec('lr', 1e-3, lambda r: np.exp(r.uniform(low=-10, high=-3))),
        HparamSpec('batch_size', 128, lambda r: int(r.choice(a=[128, 256, 512, 1024]))),

        # Network
        HparamSpec('mc_architecture', 'tcn'),
        HparamSpec('tcn_hidden_dims', 64, lambda r: r.choice(a=[32, 64, 128])),
        HparamSpec('tcn_num_layers', 1, lambda r: r.randint(low=1, high=10)),
        HparamSpec('tcn_kernel_size', 4, lambda r: r.randint(low=2, high=6)),
        HparamSpec('tcn_dropout', 0.5, lambda r: r.choice(a=[0.3, 0.4, 0.5, 0.6, 0.7]))

    ]

    def __init__(self, hparams, args):
        self.d = data.MultiCenterDataset(
            hparams['mc_outcome'], 
            self.TRAIN_PCT,
            self.VAL_PCT
        )

    def get_torch_dataset(self, envs, dset):
        datasets = []
        for r in envs:
            ds = data.SingleCenter(self.d[r][self.d[r]['fold'] == dset])
            datasets.append(ds)
        
        return ConcatDataset(datasets)

    def get_loss_fn(self):
        return bce_loss

    def get_mask(self, batch):
        _, y = batch
        return y != data.PAD_VALUE

    def get_featurizer(self, hparams):
        if hparams['mc_architecture'] == "tcn":
            return featurizer.TCNet(
                self.d.num_inputs,
                hparams['tcn_hidden_dims'],
                hparams['tcn_num_layers'],
                hparams['tcn_kernel_size'],
                hparams['tcn_dropout']
            )
        return NotImplementedError(
            f"Architecture {hparams['mc_architecture']} not available ",
            f"as a featurizer for the MultiCenter experiment"
        )

    def eval_metrics(self, algorithm, loader, device, **kwargs):
        logits, y, _ = predict_on_set(algorithm, loader, device)
        logits = logits[..., -1]

        # Mask any predictions that were made on padded values
        mask = y != data.PAD_VALUE
        logits = logits.view(-1)[mask.view(-1)].numpy()
        y = y.view(-1)[mask.view(-1)].long().numpy()

        return {
            'roc': roc_auc_score(y, logits)
        }


def _not(lst, excl):
    return [x for x in lst if x != excl]

class MultiCenterMIMIC(MultiCenterBase):
    TRAIN_ENVS = _not(MultiCenterBase.ENVIRONMENTS, 'mimic')
    VAL_ENV = 'mimic'
    TEST_ENV = 'mimic'
    
class MultiCenterEICU(MultiCenterBase):
    TRAIN_ENVS = _not(MultiCenterBase.ENVIRONMENTS, 'eicu')
    VAL_ENV = 'eicu'
    TEST_ENV = 'eicu'

class MultiCenterHIRID(MultiCenterBase):
    TRAIN_ENVS = _not(MultiCenterBase.ENVIRONMENTS, 'hirid')
    VAL_ENV = 'hirid'
    TEST_ENV = 'hirid'

class MultiCenterAUMC(MultiCenterBase):
    TRAIN_ENVS = _not(MultiCenterBase.ENVIRONMENTS, 'aumc')
    VAL_ENV = 'aumc'
    TEST_ENV = 'aumc'
