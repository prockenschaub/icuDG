import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import multiprocessing as mp
from typing import Dict, Tuple, Type

from sklearn.model_selection import KFold
from torch.utils.data import Dataset

from . import Constants

PAD_VALUE = 2


def get_cv_split(data: Dict[str, pd.DataFrame], i: int = 0, n_splits: int = 5, seed: int = 42):
    """Split the data into training, validation and test set

    Note: this function returns a single split from a repeated cross-validation.
          In order to get the full cross-validation, run with different values 
          of `i`. For example, running this function 10-times with values of 
          ranging from 0 to 9 will give 10 splits corresponding to a 2-times 
          repeated 5-fold cross-validation. 

    Args:
        data (Dict[str, pd.DataFrame]): 
        i (int, optional): _description_. Defaults to 0.
        n_splits (int, optional): _description_. Defaults to 5.
        seed (int, optional): _description_. Defaults to 42.

    Returns:
        _type_: _description_
    """
    seeds = np.random.RandomState(seed).randint(low=0, high=2**32-1, size=(2, ))
    outer = KFold(n_splits, shuffle=True, random_state=seeds[0])
    inner = KFold(n_splits, shuffle=True, random_state=seeds[1])
    
    count = 0
    all_stays = data['sta'].index
    split = {"train": {}, "val": {}, "test": {}}
    for dev, test in outer.split(all_stays):
        for train, val in inner.split(dev):
            if count == i:
                split['train']['stays'] = all_stays[dev][train]
                split['val']['stays'] = all_stays[dev][val]
                split['test']['stays'] = all_stays[test]

                for s in split.keys():    # train / val / test 
                    for d in data.keys(): # sta / dyn / outc
                        split[s][d] = data[d].loc[split[s]['stays'], :]

                break
            count += 1
    
    return split


def pad_to_len(x: np.ndarray, length: int) -> np.ndarray:
    """Pad the first dimension of an array to a given length using PAD_VALUE

    Args:
        x (np.ndarray): 2-dimensional array
        length (int): maximum length to pad. If less than `x.shape[0]`, x is shortened.

    Raises:
        ValueError: if `x` does not have exactly 2 dimensions

    Returns:
        np.ndarray: padded array
    """
    if not len(x.shape) == 2:
        raise ValueError(f'Only 2-dimensional arrays can be padded, got {len(x.shape)} dims.')
    copy_len = min(x.shape[0], length)
    x_pad = np.full((length, x.shape[1]), PAD_VALUE, dtype=np.float32)
    x_pad[:copy_len, :] = x[:copy_len, :]
    return x_pad


def pad_missing(x: np.ndarray) -> np.ndarray:
    """Pad missing values using PAD_VALUE

    Args:
        x (np.ndarray)): numpy array with missing values

    Returns:
        np.ndarray: padded array
    """
    x[np.isnan(x)] = PAD_VALUE
    return x


class Environment():
    """Data for a single ICU database (e.g., MIMIC) and outcome (e.g., mortality24)

    Args:
            db (str): data source, one of 'miiv', 'eicu', 'hirid', 'aumc'
            outcome (str): prediction target, one of 'mortality24', 'aki', 'sepsis'
    """
    def __init__(self, db, outcome, pad_to):
        self.db = db
        self.outcome = outcome
        self.pad_to = pad_to
                
    # def prepare(self, trial, n_splits, seed=42, debug=False):
    #     data = load_data(self.db, self.outcome, debug)
    #     # Hack to avoid pandas/python unnecessarily hanging on to memory. Start
    #     # subprocess which is terminated afterwards, releasing all resources. 
    #     # See https://stackoverflow.com/questions/39100971/how-do-i-release-memory-used-by-a-pandas-dataframe 
    #     data = mp.Pool(1).apply(preprocess_data, args=[data, trial, n_splits, seed])
    #     self.data = data

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
                df = df.groupby('stay_id').ffill()  # start with forward fill
                df = df.fillna(value=0)             # fill any remaining NAs with 0

                self.data[fold][part] = df

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
        return len(self.data['train']['sta'].columns) + len(self.data['train']['dyn'].columns)

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
        return Fold(self.data[fold], fold, self.pad_to)


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
        self.pats = self.data['sta'].index
        self.pad_to = pad_to

    def __len__(self):
        return len(self.pats)
    
    def __getitem__(self, idx):
        pat_id = self.pats[idx]
        
        # Get features and concatenate static and dynamic data
        X_sta = self.data['sta'].loc[pat_id].values   # D
        X_dyn = self.data['dyn'].loc[pat_id].values   # T x P
        
        num_time_steps = X_dyn.shape[0]
        X = np.concatenate((np.tile(X_sta, (num_time_steps, 1)), X_dyn), axis=1)
        X = X.astype(np.float32)

        # Get labels
        Y = self.data['outc'].loc[pat_id].values  # 1 or T x 1
        Y = Y.astype(np.float32)

        # Pad them to the right length (if necessary)
        if self.pad_to:
            X = pad_to_len(X, self.pad_to)       # pad_to x P
            if len(Y.shape) == 2:
                Y = pad_to_len(Y, self.pad_to)   # pad_to x 1
                Y = Y[:, -1] 
            Y = pad_missing(Y)

        return X, Y
