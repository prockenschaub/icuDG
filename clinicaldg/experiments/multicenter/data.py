import numpy as np
import pandas as pd
import multiprocessing as mp

from torch.utils.data import Dataset

from . import Constants

PAD_VALUE = 2
MAX_LEN = 193

def load_data(db, outcome, debug=False):
    # Get the hourly data preprocessed with the R package ``ricu``
    data = pd.read_csv(
        f'{Constants.ts_paths[db]}/{outcome}.csv', 
        index_col=['stay_id', 'time'], 
        nrows=10000 if debug else None,
        dtype=np.float32
    )
    data.rename(columns={outcome: 'label'}, inplace=True)
    return data

def preprocess_data(data, train_pct, val_pct, seed=None):
    features = data.columns[data.columns != 'label']
    
    # Randomly shuffle the patients
    pats = data.index.levels[0]
    pats = np.random.RandomState(seed).permutation(pats)
    num_pats = len(pats)
    
    # Split into train / val / test
    bounds = np.cumsum([num_pats*train_pct, num_pats*val_pct], dtype=int)
    data.loc[:, 'fold'] = ''
    data.loc[pats[:bounds[0]], 'fold'] = 'train'
    data.loc[pats[bounds[0]:bounds[1]], 'fold'] = 'val'
    data.loc[pats[bounds[1]:], 'fold'] = 'test'

    # Normalise
    means = data[data.fold == 'train'][features].mean()
    stds = data[data.fold == 'train'][features].std()
    data = pd.concat((data[['fold', 'label']], (data[features] - means) / stds), axis=1)

    # Fill missing values
    data = data.groupby('stay_id').ffill()  # start with forward fill
    data = data.fillna(value=0)             # fill any remaining NAs with 0

    return data


class Environment():
    def __init__(self, db, outcome):
        self.db = db
        self.outcome = outcome
                
    def prepare(self, train_pct=0.7, val_pct=0.1, seed=42, debug=False):
        data = load_data(self.db, self.outcome, debug)
        # Hack to avoid pandas/python unnecessarily hanging on to memory. Start
        # subprocess which is terminated afterwards, releasing all resources. 
        # See https://stackoverflow.com/questions/39100971/how-do-i-release-memory-used-by-a-pandas-dataframe 
        data = mp.Pool(1).apply(preprocess_data, args=[data, train_pct, val_pct, seed])
        self.data = data
        self.pats = data.index.get_level_values(0).unique()

    @property
    def loaded(self):
        return hasattr(self, 'data')

    def ascertain_loaded(self):
        if not self.loaded:
            raise RuntimeError(f'Data for {self.db}:{self.outcome} accessed before loading.')

    @property
    def num_inputs(self):
        self.ascertain_loaded()
        return len(self.data.drop(columns=['label', 'fold']).columns)

    def __getitem__(self, fold):
        self.ascertain_loaded()
        if not fold in ['train', 'val', 'test']:
            raise ValueError(f'fold must be one of ["train", "val", "test"], got {fold}')
        return Fold(self.data, fold)


class Fold(Dataset):
    def __init__(self, data, fold):
        self.fold = fold
        self.data = data.query(f'fold == "{fold}"').drop(['fold'], axis=1)
        self.pats = self.data.index.get_level_values(0).unique()

    def __len__(self):
        return len(self.pats)
    
    def __getitem__(self, idx):
        pat_id = self.pats[idx]
        pat_data = self.data.loc[pat_id]
        
        # Get features and labels
        num_time_steps = pat_data.shape[0]
        X = pat_data.drop('label', axis=1).values   # T x P
        Y = pat_data[['label']].values              # T x 1

        # Pad them to the right length
        X = pad_to_len(X, num_time_steps)   # MAX_LEN x P
        Y = pad_to_len(Y, num_time_steps)   # MAX_LEN x 1

        return X, Y[:, -1] 

def pad_to_len(x, len):
    x_pad = np.full((MAX_LEN, x.shape[1]), PAD_VALUE, dtype=np.float32)
    x_pad[:len, :] = x
    return x_pad
