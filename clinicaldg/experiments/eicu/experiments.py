# Based on code by Zhang et al., rearranged and refactored by 
# Patrick Rockenschaub. 

import numpy as np
import torch

from clinicaldg.lib.hparams_registry import HparamSpec
from clinicaldg.lib.evalution import binary_clf_metrics
from clinicaldg.experiments import ExperimentBase 

from . import Constants, Augmentations, featurizer
from .data import AugmentedDataset


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

    # Define hyperparameters
    HPARAM_SPEC = [
        # Training
        HparamSpec('lr', 1e-3, lambda r: 10**r.uniform(-5.0, -2.5)),
        HparamSpec('batch_size', 128),

        # Featurzier
        HparamSpec('eicu_architecture', "GRU"),
        HparamSpec('mlp_width', 128, lambda r: int(2 ** r.uniform(5, 8))),
        HparamSpec('mlp_depth', 4, lambda r: int(r.choice([2, 3, 4]))),
        HparamSpec('mlp_dropout', 0., lambda r: r.choice([0., 0.1])),
        HparamSpec('gru_hidden_dim', 128, lambda r: int(2 ** r.uniform(5, 8))),
        HparamSpec('gru_layers', 3, lambda r: int(r.choice([2, 3, 4]))),
        HparamSpec('emb_dim', 16, lambda r: int(2 ** r.uniform(3, 5))),
    ]

    def get_torch_dataset(self, envs, dset):
        return self.d.get_torch_dataset(envs, dset)

    def get_featurizer(self, hparams):
        input_size = (
            len(Constants.ts_cont_features) +
            len(Constants.ts_cat_features) * hparams['emb_dim']  + 
            len(Constants.static_cont_features) + 
            len(Constants.static_cat_features) * hparams['emb_dim']       
        )          
        
        if hparams['eicu_architecture'] == 'MLP':
            return featurizer.FlattenedDenseNet(
                input_size,
                *self.d.get_num_levels(), 
                hparams['emb_dim'], 
                hparams['mlp_depth'], 
                hparams['mlp_width'],
                dropout_p = hparams['mlp_dropout']
            )
        elif hparams['eicu_architecture'] == 'GRU':
            return featurizer.GRUNet(
                input_size,
                *self.d.get_num_levels(), 
                hparams['emb_dim'], 
                hparams['gru_layers'], 
                hparams['gru_hidden_dim'],
                dropout_p = hparams['mlp_dropout']
            )
        return NotImplementedError(
            f"Architecture {hparams['eicu_architecture']} not available ",
            f"as a featurizer for the eICU experiment"
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

    HPARAM_SPEC = eICUBase.HPARAM_SPEC + [
        HparamSpec('corr_label_train_corrupt_dist', 0.1),
        HparamSpec('corr_label_train_corrupt_mean', 0.1),
        HparamSpec('corr_label_val_corrupt', 0.5),
        HparamSpec('corr_label_test_corrupt', 0.9),
    ]

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

    HPARAM_SPEC = eICUBase.HPARAM_SPEC + [
        HparamSpec('subsample_g1_mean', 0.7),
        HparamSpec('subsample_g2_mean', 0.1),
        HparamSpec('subsample_g1_dist', 0.1),
        HparamSpec('subsample_g2_dist', 0.05),
    ]

    def __init__(self, hparams, args):
        super().__init__()
        self.d = AugmentedDataset([Augmentations.Subsample(hparams['subsample_g1_mean'], hparams['subsample_g2_mean'],
                                                hparams['subsample_g1_dist'], hparams['subsample_g2_dist'])], 
                       train_pct = eICUBase.TRAIN_PCT, val_pct = eICUBase.VAL_PCT)     
        
        
class eICUSubsampleUnobs(eICUSubsampleObs):    
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

    HPARAM_SPEC = eICUBase.HPARAM_SPEC + [
        HparamSpec('corr_noise_train_corrupt_dist', 0.5),
        HparamSpec('corr_noise_train_corrupt_mean', 2.0),
        HparamSpec('corr_noise_val_corrupt', 0.0),
        HparamSpec('corr_noise_test_corrupt', -1.0),
        HparamSpec('corr_noise_std', 0.5),
        HparamSpec('corr_noise_feature', "admissionweight"),
    ]

    def __init__(self, hparams, args):
        super().__init__()
        if hparams['corr_noise_feature'] in Constants.ts_cat_features:   # GCS Total     
            Constants.ts_cat_features.remove(hparams['corr_noise_feature'])
            Constants.ts_cont_features.append(hparams['corr_noise_feature'])
                        
        self.d = AugmentedDataset([Augmentations.GaussianNoise(hparams['corr_noise_train_corrupt_dist'], hparams['corr_noise_train_corrupt_mean'], 
                                           hparams['corr_noise_val_corrupt'], hparams['corr_noise_test_corrupt'], std = hparams['corr_noise_std'], feat_name = hparams['corr_noise_feature'])], 
                       train_pct = eICUBase.TRAIN_PCT, val_pct = eICUBase.VAL_PCT)          
        