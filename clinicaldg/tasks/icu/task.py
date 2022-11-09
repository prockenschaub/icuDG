import numpy as np
import pandas as pd
from typing import List, Dict, Callable
from functools import partial

import torch
from torch.utils.data import ConcatDataset

from clinicaldg.lib.hparams_registry import HparamSpec
from clinicaldg.lib.misc import predict_on_set, cat
from clinicaldg.lib.metrics import roc_auc_score
from clinicaldg.lib.losses import masked_bce_with_logits
from clinicaldg.tasks import base
from clinicaldg.algorithms.base import Algorithm

from . import data, featurizer


def _not(lst, excl):
    return [x for x in lst if x not in excl]


class MulticenterICU(base.Task):
    """Basic task setup for experiments using multicenter ICU data.

    Args:
        outcome (str): the outcome to predict ("mortality24", "aki", or "sepsis")
        hparams (dict): a dictionary with all relevant hyperparameters
        args (dict): additional training arguments passed via the command line

    Environments: 
        'miiv' : MIMIC IV (US)
        'eicu' : eICU (US)
        'hirid': HiRID (Switzerland)
        'aumc' : Amsterdam UMC (Netherlands)
    """

    ENVIRONMENTS = ['miiv', 'eicu', 'hirid', 'aumc']
    
    num_classes = 2
    input_shape = None
    
    HPARAM_SPEC = [
        # Data
        HparamSpec('val_env', None),
        HparamSpec('test_env', 'miiv'),

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

    def __init__(self, outcome, hparams, args):
        self.outcome = outcome
        self.args = args
        self.hparams = hparams
        self.envs = {e: data.Environment(e, outcome, self.pad_to) for e in self.ENVIRONMENTS}

        # Assign environments to train / val / test
        self.TRAIN_ENVS = _not(self.ENVIRONMENTS, [hparams['val_env']] + [hparams['test_env']])
        if hparams['val_env'] == 'train':
            self.VAL_ENVS = self.TRAIN_ENVS
        else:
            self.VAL_ENVS = [hparams['val_env']]
        self.TEST_ENVS = [hparams['test_env']]

    def setup(self, envs: List[str] = None, use_weight: bool = True) -> None:
        """Perform actual data loading and preprocessing
        
        Args:
            envs (List[str]): names of the environments to set up. Defaults to `self.ENVIRONMENTS`.
            use_weight (bool): flag indicating whether the loss should be weighed to account for class 
                class imbalance. Defaults to True.
        """

        if envs is None:
            envs = [e for e in self.envs.keys()]

        # Load and split the data
        for name, obj in self.envs.items():
            if name in envs:
                obj.load(self.args['debug'])
                obj.encode_categorical()
                obj.split(self.args['trial'], self.args['n_splits'], self.args['seed'])
        
        # Normalise and impute based on training data
        if not hasattr(self, "means") and not hasattr(self, "stds"):
            loaded = [e.db for e in self.envs.values() if e.loaded]
            if len(set(self.TRAIN_ENVS) - set(loaded)) > 0:
                raise RuntimeError(
                    f"If means and stds for are not prespecified, all training environments must be setup.",
                    f"The following training environments are missing {set(self.TRAIN_ENVS) - set(loaded)}"
                )
            train_sta = pd.concat([e.data['train']['sta'] for e in self.envs.values() if e.loaded], axis=0)
            train_dyn = pd.concat([e.data['train']['dyn'] for e in self.envs.values() if e.loaded], axis=0)
            self.means = {'sta': train_sta.mean(), 'dyn': train_dyn.mean()}
            self.stds = {'sta': train_sta.std(), 'dyn': train_dyn.std()}

        for name, obj in self.envs.items():
            if name in envs:
                obj.normalise(self.means, self.stds)
                obj.impute()

        # Check that all have the same number of inputs after preprocessing
        input_dims = np.unique([e.num_inputs for e in self.envs.values() if e.loaded])
        if len(input_dims) > 1:
            raise ValueError(f'Different input dimensions in envs: {input_dims}')
        self.num_inputs = int(input_dims)


        # Calculate case weights based on train fold of train envs
        if use_weight:
            train_data = pd.concat([self.envs[e]['train'].data['outc'] for e in self.TRAIN_ENVS])
            prop_cases = np.mean(train_data.label)
            self.case_weight = torch.tensor((1 - prop_cases) / prop_cases)
        else:
            self.case_weight = None

    def set_means_and_stds(self, means: Dict[str, pd.Series], stds: Dict[str, pd.Series]):
        """Specify means and standard devs for normalisation during setup, e.g., based on previous training

        Args:
            means (Dict[str, pd.Series]): dictionary with elements 'sta' and 'dyn' containing pd.Series
                with a mean for each columns in the static respectively dynamic data
            stds (Dict[str, pd.Series]): dictionary with elements 'sta' and 'dyn' containing pd.Series
                with a standard deviation for each columns in the static respectively dynamic data

        See also: `Environment.get_means_and_stds()`
        """
        self.means = means
        self.stds = stds

    def get_torch_dataset(self, envs: List[str], fold: str) -> ConcatDataset:
        """Get one or more envs as a torch dataset

        Args:
            envs (List[str]): list of environment names to get
            fold (str): fold to return ('train', 'val', 'test')

        Returns:
            ConcatDataset
        """
        return ConcatDataset([self.envs[e][fold] for e in envs])

    def get_featurizer(self) -> torch.nn.Module:
        """Get the torch module used to embed the preprocessed input

        Returns:
            torch.nn.Module
        """
        if self.hparams['architecture'] == "tcn":
            return featurizer.TCNet(
                self.num_inputs,
                self.hparams['hidden_dims'],
                self.hparams['num_layers'],
                self.hparams['kernel_size'],
                self.hparams['dropout']
            )
        elif self.hparams['architecture'] == "attn":
            return featurizer.TransformerNet(
                self.num_inputs,
                self.hparams['hidden_dims'],
                self.hparams['num_layers'],
                self.hparams['heads'],
                self.hparams['dropout']
            )
        return NotImplementedError(
            f"Architecture {self.hparams['architecture']} not available ",
            f"as a featurizer for the MulticenterICU task."
        )

    def get_loss_fn(self) -> Callable:
        """Return the loss function for this task, a (weighted) mask BCE loss

        Returns:
            Callable: loss function
        """
        if hasattr(self, "case_weight"):
            pos_weight = self.case_weight
        else:
            pos_weight = None
        return partial(masked_bce_with_logits, pos_weight=pos_weight)

    def eval_metrics(
        self, 
        algorithm: Algorithm, 
        loader: torch.utils.data.DataLoader, 
        device: str, 
        **kwargs
    ):
        """Calculate evaluation metrics for this task
        
        Args:
            algorithm (Algorithm): the model used for prediction
            loader (DataLoader): a data loader with the data used for evaluation
            device (str): the device on which to run the model ('cpu' or 'cuda')

        Returns:
            Dict: 
                loss: loss function on the validation data
                auroc: area under the receiver operating characteristic
        """
        logits, y, mask = predict_on_set(algorithm, loader, device, self.get_mask)
        mask = cat(mask)
        
        # Get the loss function
        loss = algorithm.loss_fn(logits, y, mask) # Loss is defined by task

        # Get the AUROC
        logits = logits[..., -1]
        logits = logits.view(-1)[mask.view(-1)].numpy()
        y = y.view(-1)[mask.view(-1)].long().numpy()
        auroc = roc_auc_score(y, logits)

        return {'loss': loss.item(), 'auroc': auroc}


class Mortality24(MulticenterICU):
    def __init__(self, hparams, args):
        self.pad_to = None
        super().__init__('mortality24', hparams, args)

    def get_mask(self, batch):
        # batch: x, y, ...
        y = batch[1]
        return torch.ones_like(y, dtype=torch.bool)

    def get_featurizer(self):
        return featurizer.LastStep(super(Mortality24, self).get_featurizer())


class AKI(MulticenterICU):
    def __init__(self, hparams, args):
        self.pad_to = 169 # one week of data
        super().__init__("aki", hparams, args)

    def get_mask(self, batch):
        # batch: x, y, ...
        y = batch[1]
        return y != data.PAD_VALUE


class Sepsis(MulticenterICU):
    def __init__(self, hparams, args):
        self.pad_to = 169 # one week of data
        super().__init__("sepsis", hparams, args)

    def get_mask(self, batch):
        # batch: x, y, ...
        y = batch[1]
        return y != data.PAD_VALUE