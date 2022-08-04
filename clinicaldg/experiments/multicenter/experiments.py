import numpy as np

import torch
from torch.utils.data import ConcatDataset

from clinicaldg import networks
from clinicaldg.lib.hparams_registry import HparamSpec
from clinicaldg.experiments import ExperimentBase 

from . import data, featurizer

class MultiCenterBase(ExperimentBase):
    
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
        HparamSpec('batch_size', 128, lambda r: r.choice(a=[128, 256, 512, 1024])),

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
        return 1 #binary_clf_metrics(preds, targets, male, env_name) # male - female
    

class MultiCenterMimic(MultiCenterBase):
    TRAIN_ENVS = [env for env in MultiCenterBase.ENVIRONMENTS if env != 'mimic']
    VAL_ENV = 'mimic'
    TEST_ENV = 'mimic'
    
