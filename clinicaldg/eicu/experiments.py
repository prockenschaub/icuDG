# Based on code by Zhang et al., rearranged and refactored by 
# Patrick Rockenschaub. 

import numpy as np

import torch

from clinicaldg.experiments import ExperimentBase 
from clinicaldg.lib.evalution import binary_clf_metrics
from clinicaldg.eicu import Constants, Augmentations, models
from clinicaldg.eicu.data import AugmentedDataset


class eICUBase(ExperimentBase):
    '''
    Base hyperparameters:
    eicu_architecture: {MLP, GRU}
    
    '''
    ENVIRONMENTS = ['Midwest', 'West', 'Northeast', 'Missing', 'South']
    TRAIN_PCT = 0.7
    VAL_PCT = 0.1
    MAX_STEPS = 2000
    N_WORKERS = 1
    CHECKPOINT_FREQ = 10
    ES_METRIC = 'roc'
    TRAIN_ENVS = ['Midwest', 'West', 'Northeast']
    VAL_ENV = 'Missing'
    TEST_ENV = 'South'
    num_classes = 2
    input_shape = None
    ES_PATIENCE = 7 # * checkpoint_freq steps

    def get_torch_dataset(self, envs, dset):
        return self.d.get_torch_dataset(envs, dset)

    def get_featurizer(self, hparams):
        if hparams['eicu_architecture'] == 'MLP':
            return models.FlattenedDense(
                *self.d.get_num_levels(), 
                hparams['emb_dim'], 
                hparams['mlp_depth'], 
                hparams['mlp_width'],
                dropout_p = hparams['mlp_dropout']
            )
        elif hparams['eicu_architecture'] == 'GRU':
            return models.GRUNet(
                *self.d.get_num_levels(), 
                hparams['emb_dim'], 
                hparams['gru_layers'], 
                hparams['gru_hidden_dim'],
                dropout_p = hparams['mlp_dropout']
            )

    def predict_on_set(self, algorithm, loader, device):
        preds, targets, genders = [], [], []
        with torch.no_grad():
            for x, y in loader:
                x = {j: x[j].to(device) for j in x}
                algorithm.eval()
                logits = algorithm.predict(x)

                targets += y.detach().cpu().numpy().tolist()
                genders += x['gender'].cpu().numpy().tolist()
                preds_list = torch.nn.Softmax(dim = 1)(logits)[:, 1].detach().cpu().numpy().tolist()
                if isinstance(preds_list, list):
                    preds += preds_list
                else:
                    preds += [preds_list]
        return np.array(preds), np.array(targets), np.array(genders)

    def eval_metrics(self, algorithm, loader, env_name, weights, device):
        preds, targets, genders = self.predict_on_set(algorithm, loader, device)
        male = genders == 1
        return binary_clf_metrics(preds, targets, male, env_name) # male - female
    

class eICU(eICUBase): 
    def __init__(self, hparams, args):
        super().__init__()
        self.d = AugmentedDataset([], train_pct = eICUBase.TRAIN_PCT, 
                                           val_pct = eICUBase.VAL_PCT)   
        
        
class eICUCorrLabel(eICUBase):    
    '''
    Hyperparameters:
    corr_label_train_corrupt_dist
    corr_label_train_corrupt_mean
    corr_label_val_corrupt
    corr_label_test_corrupt
    '''
    def __init__(self, hparams, args):
        super().__init__()
        self.d = AugmentedDataset([Augmentations.AddCorrelatedFeature(hparams['corr_label_train_corrupt_dist'], 
                              hparams['corr_label_train_corrupt_mean'], hparams['corr_label_val_corrupt'], 
                              hparams['corr_label_test_corrupt'], 'corr_label')], 
                       train_pct = eICUBase.TRAIN_PCT, val_pct = eICUBase.VAL_PCT)  
        
        Constants.static_cont_features.append('corr_label')
        

class eICUSubsampleObs(eICUBase):    
    '''
    Hyperparameters:
    subsample_g1_mean
    subsample_g2_mean
    subsample_g1_dist
    subsample_g2_dist
    '''
    def __init__(self, hparams, args):
        super().__init__()
        self.d = AugmentedDataset([Augmentations.Subsample(hparams['subsample_g1_mean'], hparams['subsample_g2_mean'],
                                                hparams['subsample_g1_dist'], hparams['subsample_g2_dist'])], 
                       train_pct = eICUBase.TRAIN_PCT, val_pct = eICUBase.VAL_PCT)     
        
        
class eICUSubsampleUnobs(eICUSubsampleObs):    
    '''
    Hyperparameters:
    subsample_g1_mean
    subsample_g2_mean
    subsample_g1_dist
    subsample_g2_dist
    '''
    def __init__(self, hparams, args):
        Constants.static_cat_features.remove('gender')
        super().__init__(hparams, args)
                    
        
class eICUCorrNoise(eICUBase):    
    '''
    Hyperparameters:
    corr_noise_train_corrupt_dist
    corr_noise_train_corrupt_mean
    corr_noise_val_corrupt
    corr_noise_test_corrupt
    corr_noise_std
    corr_noise_feature
    '''
    def __init__(self, hparams, args):
        super().__init__()
        if hparams['corr_noise_feature'] in Constants.ts_cat_features:   # GCS Total     
            Constants.ts_cat_features.remove(hparams['corr_noise_feature'])
            Constants.ts_cont_features.append(hparams['corr_noise_feature'])
                        
        self.d = AugmentedDataset([Augmentations.GaussianNoise(hparams['corr_noise_train_corrupt_dist'], hparams['corr_noise_train_corrupt_mean'], 
                                           hparams['corr_noise_val_corrupt'], hparams['corr_noise_test_corrupt'], std = hparams['corr_noise_std'], feat_name = hparams['corr_noise_feature'])], 
                       train_pct = eICUBase.TRAIN_PCT, val_pct = eICUBase.VAL_PCT)          
        