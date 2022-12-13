import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import multiprocessing as mp
from typing import Dict, Tuple, Type

import torch
from torch.utils.data import Dataset, TensorDataset

from ...lib.resampling import get_cv_split
from ...lib.misc import pad_to_len, pad_missing
from . import Constants

PAD_VALUE = 2


class ICUEnvironment():
    """Data for a single ICU database (e.g., MIMIC) and outcome (e.g., mortality24)

    Args:
            db (str): data source, one of 'miiv', 'eicu', 'hirid', 'aumc'
            outcome (str): prediction target, one of 'mortality24', 'aki', 'sepsis'
    """
    def __init__(self, db, outcome, pad_to):
        self.db = db
        self.outcome = outcome
        self.pad_to = pad_to

    def load(self, debug=False) -> None:
        """Load hourly data processed with the R package ``ricu``

        Args:
            db (str): data source, one of 'miiv', 'eicu', 'hirid', 'aumc'
            outcome (str): prediction target, one of 'mortality24', 'aki', 'sepsis'
            debug (bool, optional): flag to load only a subset of 1000 patients for debugging. Defaults to False.

        Returns:
            Dict[str, pd.DataFrame]: 
                'sta' : static data like age or sex
                'dyn' : time-varying data like heart rate or blood pressure
                'outc': the outcome
        """
        data = {}
        for part in ["sta", "dyn", "outc"]:
            tbl = pq.read_table(f'{Constants.data_dir}/{self.outcome}/{self.db}/{part}.parquet')
            tbl = tbl.to_pandas()
            if "time" in tbl.columns:
                tbl = tbl.set_index(['stay_id', 'time'])
            else:
                tbl = tbl.set_index(['stay_id'])
            data[part] = tbl

        if debug:
            # Limit to 1000 patients
            debug_stays = data['sta'].index.values[:1000]
            data = {k: v.loc[debug_stays, :] for k, v in data.items()}

        self.data = data

    def encode_categorical(self) -> None:
        """Encode any categorical variables (only sex in the current version)"""
        self.ascertain_loaded()
        self.data['sta']['sex'] = (self.data['sta']['sex'] == "Female").astype(float)

    def split(self, trial, n_splits, seed=None):
        """Split the data into training, validation, and test sets

        See also: `get_cv_split()`
        """
        self.ascertain_loaded()
        self.data = mp.Pool(1).apply(get_cv_split, args=[self.data, trial, n_splits, seed])

    def get_means_and_stds(self, fold='train') -> Tuple[Dict[str, pd.Series]]:
        """Calculate means and standard deviations for static and dynamic predictors

        Args:
            fold (str, optional): Fold for which to calculate. Defaults to 'train'.

        Returns:
            Tuple[Dict[str, pd.Series]]: means and standard deviations for static
                ('sta') and dynamic ('dyn) predictors
        """
        means, stds = {}, {}
        for part in ['sta', 'dyn']:
            means[part] = self.data[fold][part].mean(numeric_only=True)
            stds[part] = self.data[fold][part].std(numeric_only=True)
        return means, stds

    def normalise(self, means: Dict[str, pd.Series] = None, stds=None) -> None:
        """Normalise the predictors using mean and standard deviation

        Args:
            means (Dict[str, pd.Series], optional): Means of static('sta') and dynamic ('dyn) predictors. 
                Defaults to means of own training fold.
            stds (Dict[str, pd.Series], optional): Standard deviation of static('sta') and dynamic ('dyn) 
                predictors. Defaults to standard deviations of own training fold.
        """
        if means is None and stds is None:
            # Calculate metrics from training split if they aren't provided
            means, stds = self.get_means_and_stds()
        
        for part in ['sta', 'dyn']:
            # Apply processsing to all folds using pre-calculated metrics where needed
            for fold in ['train', 'val', 'test']:
               self.data[fold][part] = (self.data[fold][part] - means[part]) / stds[part]
    
    def impute(self) -> None:
        """Impute data with forward fill.

        Note: Time steps without a prior entry are filled with 0. If applied after normalisation, this 
            corresponds to an average value in the ICU dataset.
        """
        for part in ['sta', 'dyn']:
            # Apply processsing to all folds using pre-calculated metrics where needed
            for fold in ['train', 'val', 'test']:
                df = self.data[fold][part]

                # Add indicators for missingness
                inds = df.isna().astype(float)
                inds.columns = [name+"_ind" for name in inds.columns]
                df = pd.concat((df, inds), axis=1) 

                # Fill missing values
                df = df.groupby(level=0).ffill()  # start with forward fill
                df = df.fillna(value=0)             # fill any remaining NAs with 0

                self.data[fold][part] = df

    def to_X_y(self, pad_to=None) -> None:
        for fold in ['train', 'val', 'test']:
            self.data[fold] = fold_to_torch(self.data[fold], pad_to=pad_to)

    @property
    def loaded(self) -> bool:
        """Has the environment data been loaded yet?"""
        return hasattr(self, 'data')

    @property
    def splits(self) -> Dict[str, pd.Index]:
        return {f: v['sta'].index for f, v in self.data.items()}

    def ascertain_loaded(self):
        """Raise an error if data has not been loaded yet."""
        if not self.loaded:
            raise RuntimeError(f'Attempted to access data from {self.db}:{self.outcome} before its loaded.')

    @property
    def num_inputs(self) -> int:
        """combined number of inputs across static and dynamic predictors"""
        self.ascertain_loaded()
        if isinstance(self.data['train'], dict):
            return len(self.data['train']['sta'].columns) + len(self.data['train']['dyn'].columns)
        elif isinstance(self.data['train'], tuple):
            return self.data['train'][0][0].shape[1]

    def __getitem__(self, fold: str) -> Type["Fold"]:
        """Training, validation, or test fold of the current environment
        
        Args:
            fold (str): type of fold to return, one of 'train', 'val', or 'test'.

        Returns:
            Fold: pytorch dataset representing a training, validation, or test fold.
        """
        self.ascertain_loaded()
        if not fold in ['train', 'val', 'test']:
            raise ValueError(f'fold must be one of ["train", "val", "test"], got {fold}')
        return TensorDataset(*self.data[fold])



def fold_to_torch(data, pad_to=None):
    pats = data['sta'].index
    
    inputs, targets = [], []
    for pat_id in pats:
        # Get features and concatenate static and dynamic data
        X_sta = data['sta'].loc[pat_id].values   # D
        X_dyn = data['dyn'].loc[pat_id].values   # T x P
        
        num_time_steps = X_dyn.shape[0]
        X = np.concatenate((np.tile(X_sta, (num_time_steps, 1)), X_dyn), axis=1)
        X = X.astype(np.float32)

        # Get labels
        Y = data['outc'].loc[pat_id].values

        # Pad them to the right length (if necessary)
        if pad_to:
            X = pad_to_len(X, pad_to, PAD_VALUE)       # pad_to x P
            if len(Y.shape) == 2:
                Y = pad_to_len(Y, pad_to, PAD_VALUE)   # pad_to x 1
                Y = Y[:, -1] 
            Y = pad_missing(Y)
        
        Y = Y.astype(np.int64)

        inputs.append(X)
        targets.append(Y)
    
    return torch.tensor(np.stack(inputs)), torch.tensor(np.stack(targets))


class Fold(Dataset):
    """A single training, validation, or test fold of an environment

    Args:
        data (Dict[str, pd.DataFrame]): fold as dictionary of DataFrames, see also `Environment.load()`
        fold (str): type of fold represented by this object, one of 'train', 'val', or 'test'.
        pad_to (int): maximum number of time steps that dynamic ('dyn') predictors should be padded to. 
    """
    def __init__(self, data: Dict[str, pd.DataFrame], fold: str, pad_to: int = None):
        self.fold = fold
        self.data = data

    def __len__(self):
        return len(self.data[0])
    
    def __getitem__(self, idx):
        X, Y = self.data[0][idx], self.data[1][idx]

        return X, Y
