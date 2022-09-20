import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, ConcatDataset

from clinicaldg.lib.hparams_registry import HparamSpec
from clinicaldg.lib.misc import predict_on_set
from clinicaldg.lib.metrics import roc_auc_score
from clinicaldg.experiments import base
from clinicaldg.networks import MLP

from clinicaldg.experiments.augmentation_utils import corrupt
from .data import SyntheticDataset


def bce_loss(logits, y, mask, reduction='mean', pos_weight=None, **kwargs):
    logits = logits[..., -1]
    if pos_weight is not None:
        pos_weight = y.new_tensor(pos_weight)

    return F.binary_cross_entropy_with_logits(
        logits, 
        y, 
        reduction=reduction, 
        pos_weight=pos_weight,
        **kwargs
    )

class Synthetic(base.Experiment):
    '''
    Hyperparameters:
    cmnist_eta
    cmnist_beta
    cmnist_delta   
    
    '''
    TRAIN_PCT = 0.7
    VAL_PCT = 0.1
    MAX_STEPS = 2000
    N_WORKERS = 1
    CHECKPOINT_FREQ = 10
    num_classes = 2
    ES_PATIENCE = 20
    ES_METRIC = 'roc'
    
    # Define hyperparameters
    HPARAM_SPEC = [
        HparamSpec('num_envs', 10),
        HparamSpec('num_smpls', 5000),
        HparamSpec('val_env', 8),
        HparamSpec('test_env', 9),
        
        # Training
        HparamSpec('lr', 1e-3, lambda r: 10**r.uniform(-4.5, -2.5)),
        HparamSpec('batch_size', 128, lambda r: int(2**r.uniform(6, 9))),

        # Featurzier
        HparamSpec('mlp_width', 8, lambda r: int(2 ** r.uniform(6, 10))),
        HparamSpec('mlp_depth', 2, lambda r: int(r.choice([2, 3]))),
        HparamSpec('mlp_dropout', 0., lambda r: r.choice([0., 0.1]))
    ]

    def __init__(self, hparams, args):
        super().__init__()

        self.ENVIRONMENTS = [i for i in range(hparams['num_envs'])]
        self.TRAIN_ENVS = [i for i in self.ENVIRONMENTS if i not in [hparams['val_env'], hparams['test_env']]]
        self.VAL_ENVS = [hparams['val_env']]
        self.TEST_ENVS = [hparams['test_env']]

        self.d = SyntheticDataset(hparams['num_envs'], hparams['num_smpls'], args.seed)   
        self.input_shape = self.d[0].shape[1] - 1
    
    def get_torch_dataset(self, envs, dset):
        '''
        envs: a list of region names
        dset: split within envs, one of ['train', 'val', 'test']
        '''
        datasets = []
        
        for e in envs:
            data = self.d[e]
            
            if dset == 'train':
                idx_start, idx_end = 0, int(len(data) * self.TRAIN_PCT)
            elif dset == 'val':
                idx_start, idx_end = int(len(data) * self.TRAIN_PCT), int(len(data) * (self.TRAIN_PCT + self.VAL_PCT))
            elif dset == 'test':
                idx_start, idx_end = int(len(data) * (self.TRAIN_PCT + self.VAL_PCT)), len(data)
            else:
                raise NotImplementedError
                    
            datasets.append(TensorDataset(
                torch.tensor(data[idx_start:idx_end, 1:]), 
                torch.tensor(data[idx_start:idx_end, :1]).squeeze(-1)
            )) 
            
        return ConcatDataset(datasets) 

    def get_loss_fn(self):
        return bce_loss

    def get_mask(self, batch):
        x,y = batch
        return torch.ones_like(y)

    def get_featurizer(self, hparams):
        return MLP(self.input_shape, hparams['mlp_width'], hparams['mlp_depth'], hparams['mlp_width'], hparams['mlp_dropout'])

    def eval_metrics(self, algorithm, loader, device, **kwargs):
        logits, y, _ = predict_on_set(algorithm, loader, device)
        logits = logits[..., -1]
        return {'roc': roc_auc_score(y, logits)}



class SyntheticCorrLabel(Synthetic):
    def __init__(self, hparams, args):
        super().__init__(hparams, args)
        self.add_corrlabel(hparams, args.seed)
        self.input_shape = self.d[0].shape[1] - 1

    def add_corrlabel(self, hparams, seed):
        rand_state = np.random.RandomState(seed)
        for i in range(len(self.d)):
            if i in self.TRAIN_ENVS:
                p = rand_state.rand(1) * 0.3
            elif i in self.VAL_ENVS:
                p = 0.5
            else:
                p = 0.9
            
            c = corrupt(self.d[i][:, 0], p)
            self.d.ds[i] = np.concatenate((self.d[i], c[:, None]), axis=1)


class SyntheticEnvLabel(Synthetic):
    def __init__(self, hparams, args):
        super().__init__(hparams, args)
        self.add_envlabel(hparams, args.seed)
        self.input_shape = self.d[0].shape[1] - 1

    def add_envlabel(self, hparams, seed):
        rand_state = np.random.RandomState(seed)
        for i in range(len(self.d)):
            for j in range(len(self.d)):
                if j in self.TRAIN_ENVS:
                    if i == j:
                        c = corrupt(self.d[i][:, 0], 0.2)
                    else: 
                        c = rand_state.permutation(self.d[i][:, 0])
            
                    self.d.ds[i] = np.concatenate((self.d[i], c[:, None]), axis=1)

class SyntheticEnvCorrLabel(Synthetic):
    def __init__(self, hparams, args):
        super().__init__(hparams, args)
        self.add_envcorrlabel(hparams, args.seed)
        self.input_shape = self.d[0].shape[1] - 1

    def add_envcorrlabel(self, hparams, seed):
        rand_state = np.random.RandomState(seed)
        for i in range(len(self.d)):
            for j in range(len(self.d)):
                if j in self.TRAIN_ENVS:
                    if i == j:
                        ind = np.ones_like(self.d[i][:, 0])
                        c = corrupt(self.d[i][:, 0], 0)
                    else: 
                        ind = np.zeros_like(self.d[i][:, 0])
                        c = rand_state.permutation(self.d[i][:, 0])
            
                    self.d.ds[i] = np.concatenate((self.d[i], ind[:, None], c[:, None]), axis=1)

