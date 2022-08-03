# Based on code by Zhang et al., rearranged and refactored by 
# Patrick Rockenschaub. 

import numpy as np
import pandas as pd

import torch

from sklearn.metrics import roc_auc_score, recall_score

from clinicaldg.experiments import ExperimentBase 
from clinicaldg.lib import misc
from clinicaldg.lib.evalution import binary_clf_metrics, compute_opt_thres
from clinicaldg.cxr import Constants, Augmentations, process, models
from clinicaldg.cxr.data import get_dataset, GenderConcatDataset

class CXRBase(ExperimentBase):
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
    VAL_ENV = 'NIH'
    TEST_ENV = 'PAD'    
    NUM_SAMPLES_VAL = 1024*8 # use a subset of the validation set for early stopping
    
    def __init__(self, hparams, args):
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

    def get_featurizer(self, hparams):
        return models.EmbModel('densenet', pretrain = True, concat_features = 0)

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

    def multilabel_metrics(self, preds, targets, male, prefix, suffix, thress = None):
        if thress is None:
            thress = [0.5] * self.num_classes
            
        tpr_male = np.mean([recall_score(targets[:, i][male], preds[:, i][male] >= thress[i], zero_division=0) for i in range(self.num_classes)])
        tpr_female = np.mean([recall_score(targets[:, i][~male], preds[:, i][~male] >= thress[i], zero_division=0) for i in range(self.num_classes)])
        prev_male = np.mean([(preds[:, i][male] >= thress[i]).sum() / male.sum()  for i in range(self.num_classes)])
        prev_female = np.mean([(preds[:, i][~male] >= thress[i]).sum() / (~male).sum()  for i in range(self.num_classes)])    
        
        return {prefix + 'tpr_gap' + suffix: tpr_male - tpr_female, prefix + 'parity_gap'+ suffix: prev_male - prev_female}
    
    def eval_metrics(self, algorithm, loader, env_name, weights, device):
        preds, targets, genders = self.predict_on_set(algorithm, loader, device)
        male = genders == 'M'
        
        roc = np.mean([roc_auc_score(targets[:, i], preds[:, i]) for i in range(self.num_classes)])
        results = self.multilabel_metrics(preds, targets, male, prefix = env_name+ '_', suffix = '')
        results[env_name+'_roc'] = roc
        
        opt_thress = [compute_opt_thres(targets[:, i], preds[:, i])  for i in range(self.num_classes)]
        results.update(self.multilabel_metrics(preds, targets, male, prefix = env_name + '_', suffix = '_thres', thress = opt_thress))
                
        return results    
    
      
class CXRBinary(CXRBase):
    num_classes = 2

    def get_torch_dataset(self, envs, dset):
        augment = 0 if dset in ['val', 'test'] else self.hparams['cxr_augment']    
        return get_dataset(self.dfs, envs = envs, split = dset, imagenet_norm = True, only_frontal = True, augment = augment, cache = self.use_cache ,
                                  subset_label = 'Pneumonia')

    def eval_metrics(self, algorithm, loader, env_name, weights, device):
        preds, targets, genders = self.predict_on_set(algorithm, loader, device)
        male = genders == 'M'        
        return binary_clf_metrics(preds, targets, male, env_name)
       

class CXRSubsampleUnobs(CXRBinary):
    '''
    Hyperparameters:
    subsample_g1_mean
    subsample_g2_mean
    subsample_g1_dist
    subsample_g2_dist
    '''
    
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
        return models.EmbModel('densenet', pretrain = True, concat_features = 1)
