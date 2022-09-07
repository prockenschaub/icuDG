# Based on code by Zhang et al., rearranged and refactored by 
# Patrick Rockenschaub. 

import warnings
import numpy as np
import pandas as pd

import torch
from sklearn.metrics import roc_auc_score, recall_score

from clinicaldg.lib import misc
from clinicaldg.lib.hparams_registry import HparamSpec
from clinicaldg.lib.metrics import cross_entropy
from clinicaldg.lib.metrics import (
    compute_opt_thres, 
    roc_auc_score,
    accuracy_score,
    recall_score, 
    tnr, 
    matthews_corrcoef
)
from clinicaldg.experiments import base 

from . import Constants, Augmentations, process, featurizer
from .data import get_dataset, GenderConcatDataset

def extract_gender(batch):
    _, _, meta = batch
    return meta['Sex']

def binary_metrics(preds, targets, grp):
    thres = compute_opt_thres(targets, preds)
    preds_rounded_opt = (preds >= thres)
    tpr_gap_opt = recall_score(targets[grp], preds_rounded_opt[grp], zero_division = 0) - recall_score(targets[~grp], preds_rounded_opt[~grp], zero_division = 0)
    tnr_gap_opt = tnr(targets[grp], preds_rounded_opt[grp]) - tnr(targets[~grp], preds_rounded_opt[~grp])
    parity_gap_opt = (preds_rounded_opt[grp].sum() / grp.sum()) - (preds_rounded_opt[~grp].sum() / (~grp).sum())    
    phi_opt = matthews_corrcoef(preds_rounded_opt, grp)
    
    return {
        'roc': roc_auc_score(targets, preds),
        'acc': accuracy_score(targets, preds_rounded_opt),
        'tpr_gap': tpr_gap_opt,
        'tnr_gap': tnr_gap_opt,
        'parity_gap': parity_gap_opt,
        'phi': phi_opt
    }

def multilabel_metrics(preds, targets, grp, num_classes):
    roc = np.mean([roc_auc_score(targets[:, i], preds[:, i]) for i in range(num_classes)])
    thres = [compute_opt_thres(targets[:, i], preds[:, i])  for i in range(num_classes)]
    tpr_male = np.mean([recall_score(targets[:, i][grp], preds[:, i][grp] >= thres[i], zero_division=0) for i in range(num_classes)])
    tpr_female = np.mean([recall_score(targets[:, i][~grp], preds[:, i][~grp] >= thres[i], zero_division=0) for i in range(num_classes)])
    prev_male = np.mean([(preds[:, i][grp] >= thres[i]).sum() / grp.sum()  for i in range(num_classes)])
    prev_female = np.mean([(preds[:, i][~grp] >= thres[i]).sum() / (~grp).sum()  for i in range(num_classes)])    
    
    return {
        'roc': roc,
        'tpr_gap': tpr_male - tpr_female, 
        'parity_gap': prev_male - prev_female
    }


class CXRBase(base.Experiment):
    '''
    Base hyperparameters:
    cxr_augment: {0, 1}
    use_cache: {0, 1}
    '''
    ENVIRONMENTS = ['MIMIC', 'CXP', 'NIH', 'PAD']
    MAX_STEPS = 20000
    N_WORKERS = 1
    CHECKPOINT_FREQ = 100
    ES_METRIC = 'roc'
    input_shape = None
    ES_PATIENCE = 5 #  * checkpoint_freq steps
    TRAIN_ENVS = ['MIMIC', 'CXP']
    VAL_ENVS = ['NIH']
    TEST_ENVS = ['PAD']    
    NUM_SAMPLES_VAL = 1024*8 # use a subset of the validation set for early stopping
    
    # Define hyperparameters
    HPARAM_SPEC = [
        # Data
        HparamSpec('cxr_augment', 1),
        HparamSpec('use_cache', 0),

        # Training
        HparamSpec('lr', 5e-4, lambda r: 10**r.uniform(-5.0, -2.5)),
        HparamSpec('batch_size', 12),
    ]

    def __init__(self, hparams, args):
        warnings.warn(
            "All experiments relating to Chest X-rays were refactored in this "
            "repository to align with other experiments, but where not tested. "
            "It is likely that errors occurred when refactoring. Please refer "
            "to the original code in https://github.com/MLforHealth/ClinicalDG"
        )
        
        self.hparams = hparams
        self.use_cache = bool(self.hparams['use_cache']) if 'use_cache' in self.hparams else False

        # loads data with random splits
        self.dfs = {}
        for env in Constants.df_paths:
            func = process.get_process_func(env)
            df_env = func(pd.read_csv(Constants.df_paths[env]), only_frontal = True)
            train_df, valid_df, test_df = process.split(df_env)
            self.dfs[env] = {
                'train': train_df,
                'val': valid_df,
                'test': test_df
            }

    def get_loss_fn(self):
        return cross_entropy  # NOTE: not tested after refactor

    def get_mask(self, batch):
        _, y = batch
        return torch.ones_like(y)

    def get_featurizer(self, hparams):
        return featurizer.EmbNet('densenet', pretrain=True, concat_features=0)

    def predict_on_set(self, algorithm, loader, device):
        preds, targets, genders = [], [], []
        with torch.no_grad():
            for x, y, meta in loader:
                x = misc.to_device(x, device)
                algorithm.eval()
                logits = algorithm.predict(x)

                targets += y.detach().cpu().numpy().tolist()
                genders += meta['Sex']
                if y.ndim == 1 or y.shape[1] == 1: # multiclass
                    preds_list = torch.nn.Softmax(dim = 1)(logits)[:, 1].detach().cpu().numpy().tolist()
                else: # multilabel
                    preds_list = torch.sigmoid(logits).detach().cpu().numpy().tolist()
                if isinstance(preds_list, list):
                    preds += preds_list
                else:
                    preds += [preds_list]
        return np.array(preds), np.array(targets), np.array(genders) 
        
class CXR(CXRBase):    
    num_classes = len(Constants.take_labels)   

    def get_torch_dataset(self, envs, dset):
        augment = 0 if dset in ['val', 'test'] else self.hparams['cxr_augment']            
        return get_dataset(self.dfs, envs = envs, split = dset, imagenet_norm = True, only_frontal = True, augment = augment, cache = self.use_cache)
    
    def eval_metrics(self, algorithm, loader, device, **kwargs):
        preds, targets, genders = misc.predict_on_set(algorithm, loader, device, extract_gender)
        
        # Post-process predictions for evaluation
        if targets.ndim == 1 or targets.shape[1] == 1: # multiclass
            preds = torch.softmax(preds, dim=1)[:, 1].numpy()
        else: # multilabel
            preds = torch.sigmoid(preds).numpy()
        targets = targets.numpy()
        genders = misc.cat(genders).numpy()
        male = genders == 'M'
        
        # Calculate metrics 
        results = self.multilabel_metrics(preds, targets, male, self.num_classes)
        
        return results    
    
      
class CXRBinary(CXRBase):
    num_classes = 2

    def get_torch_dataset(self, envs, dset):
        augment = 0 if dset in ['val', 'test'] else self.hparams['cxr_augment']    
        return get_dataset(self.dfs, envs = envs, split = dset, imagenet_norm = True, only_frontal = True, augment = augment, cache = self.use_cache ,
                                  subset_label = 'Pneumonia')

    def eval_metrics(self, algorithm, loader, device, **kwargs):
        preds, targets, genders = misc.predict_on_set(algorithm, loader, device, extract_gender)

        # Post-process predictions for evaluation
        preds = torch.softmax(preds, dim=1)[:, 1].numpy()
        targets = targets.numpy()
        genders = misc.cat(genders).numpy()
        male = genders == 'M'        

        # Calculate the metrics
        results = binary_metrics(preds, targets, male)

        return results
       

class CXRSubsampleUnobs(CXRBinary):
    '''
    Hyperparameters:
    subsample_g1_mean
    subsample_g2_mean
    subsample_g1_dist
    subsample_g2_dist
    '''
    
    HPARAM_SPEC = CXRBase.HPARAM_SPEC + [
        HparamSpec('subsample_g1_mean', 0.15),
        HparamSpec('subsample_g2_mean', 0.025),
        HparamSpec('subsample_g1_dist', 0.1),
        HparamSpec('subsample_g2_dist', 0.01),
    ]

    def __init__(self, hparams, args):
        super().__init__(hparams, args)
        self.dfs = Augmentations.subsample_augment(self.dfs, hparams['subsample_g1_mean'], 
                                                                   hparams['subsample_g2_mean'], hparams['subsample_g1_dist'], hparams['subsample_g2_dist'])
        
    def get_torch_dataset(self, envs, dset):
        augment = 0 if dset in ['val', 'test'] else self.hparams['cxr_augment']    
        return get_dataset(self.dfs, envs = envs, split = dset, imagenet_norm = True, only_frontal = True, augment = augment, cache = self.use_cache,
                                  subset_label = 'Pneumonia')   
    
    
class CXRSubsampleObs(CXRSubsampleUnobs):        
    def get_torch_dataset(self, envs, dset):
        augment = 0 if dset in ['val', 'test'] else self.hparams['cxr_augment']    
        return GenderConcatDataset(get_dataset(self.dfs, envs = envs, split = dset, 
                                  imagenet_norm = True, only_frontal = True, augment = augment, cache = self.use_cache,
                                  subset_label = 'Pneumonia'))    

    def get_featurizer(self, hparams):
        return featurizer.EmbNet('densenet', pretrain=True, concat_features=1)
