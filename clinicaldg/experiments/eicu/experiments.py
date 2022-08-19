# Based on code by Zhang et al., rearranged and refactored by 
# Patrick Rockenschaub. 

import torch
import torch.nn.functional as F

from clinicaldg.lib.hparams_registry import HparamSpec
from clinicaldg.lib.metrics import (
    compute_opt_thres, 
    roc_auc_score,
    accuracy_score,
    recall_score, 
    tnr, 
    matthews_corrcoef
)
from clinicaldg.experiments import base 
from clinicaldg.lib.misc import predict_on_set, cat

from . import Constants, Augmentations, featurizer
from .data import AugmentedDataset


class eICUBase(base.Experiment):
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

    def get_loss_fn(self):
        return F.cross_entropy

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

    def eval_metrics(self, algorithm, loader, device, **kwargs):
        def extract_gender(batch):
            return batch[0]['gender']
        
        preds, targets, genders = predict_on_set(algorithm, loader, device, extract_gender)
        
        # Post-process predictions for evaluation
        preds = F.softmax(preds, dim=1)[:, 1].numpy()
        targets = targets.numpy()
        genders = cat(genders).numpy()
        male = genders == 1

        # Calculate metrics
        opt_thres = compute_opt_thres(targets, preds)

        preds_rounded_opt = (preds >= opt_thres)
        tpr_gap_opt = recall_score(targets[male], preds_rounded_opt[male], zero_division = 0) - recall_score(targets[~male], preds_rounded_opt[~male], zero_division = 0)
        tnr_gap_opt = tnr(targets[male], preds_rounded_opt[male]) - tnr(targets[~male], preds_rounded_opt[~male])
        parity_gap_opt = (preds_rounded_opt[male].sum() / male.sum()) - (preds_rounded_opt[~male].sum() / (~male).sum())    
        phi_opt = matthews_corrcoef(preds_rounded_opt, male)

        return {
            'roc': roc_auc_score(targets, preds),
            'acc': accuracy_score(targets, preds_rounded_opt),
            'tpr_gap': tpr_gap_opt,
            'tnr_gap': tnr_gap_opt,
            'parity_gap': parity_gap_opt,
            'phi': phi_opt
        }


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
        