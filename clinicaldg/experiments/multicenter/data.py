import numpy as np
import pandas as pd

from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler

from . import Constants

PAD_VALUE = 2
MAX_LEN = 193

def count_features(df):
    return len(df.drop(columns=['label', 'fold']).columns)


class MultiCenterDataset():
    def __init__(self, outcome='sepsis', train_pct = 0.7, val_pct = 0.1, seed=None):        
        self.dfs = {
            db: MultiCenterDataset.prepare_dataset(db, outcome, train_pct, val_pct, seed)
            for db in Constants.ts_paths.keys()
        }

        num_features = [count_features(df) for df in self.dfs.values()]
        assert(len(np.unique(num_features)) == 1)
    
    @property
    def num_inputs(self):
        return count_features(list(self.dfs.values())[0])

    def __getitem__(self, idx):
        return self.dfs[idx]

    def prepare_dataset(db, outcome, train_pct, val_pct, seed=None):
        """_summary_

        Args:
            db (_type_): _description_
            outcome (_type_): _description_
            train_pct (_type_): _description_
            val_pct (_type_): _description_

        Returns:
            _type_: _description_
        """
        # Get the hourly data preprocessed with the R package ``ricu``
        path = f'{Constants.ts_paths[db]}/{outcome}.csv'
        df = pd.read_csv(path, index_col=['stay_id', 'time'])
        df.rename(columns={outcome: 'label'}, inplace=True)
        features = df.columns[df.columns != 'label']

        # Randomly shuffle the patients
        pats = df.index.levels[0]
        pats = np.random.RandomState(seed).permutation(pats)
        num_pats = len(pats)
        
        # Split into train / val / test
        bounds = np.cumsum([num_pats*train_pct, num_pats*val_pct], dtype=int)
        df.loc[:, 'fold'] = ''
        df.loc[pats[:bounds[0]], 'fold'] = 'train'
        df.loc[pats[bounds[0]:bounds[1]], 'fold'] = 'val'
        df.loc[pats[bounds[1]:], 'fold'] = 'test'

        # Normalise
        means = df[df.fold == 'train'][features].mean()
        stds = df[df.fold == 'train'][features].std()
        df = pd.concat((df[['fold', 'label']], (df[features] - means) / stds), axis=1)

        # Fill missing values
        df = df.groupby('stay_id').ffill()  # start with forward fill
        df = df.fillna(value=0)             # fill any remaining NAs with 0

        return df
           
class SingleCenter(Dataset):
    def __init__(self, df):
        self.fold = df['fold'].unique()
        self.df = df.drop(['fold'], axis=1)
        self.pats = df.index.get_level_values(0).unique()
    
    def __len__(self):
        return len(self.pats)
    
    def __getitem__(self, idx):
        pat_id = self.pats[idx]
        pat_data = self.df.loc[pat_id]
        
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
