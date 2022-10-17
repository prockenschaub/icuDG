import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import multiprocessing as mp

from sklearn.model_selection import KFold

from torch.utils.data import Dataset

from . import Constants

PAD_VALUE = 2


def load_data(db, outcome, debug=False):
    # Get the hourly data preprocessed with the R package ``ricu``
    data = {}
    for part in ["sta", "dyn", "outc"]:
        tbl = pq.read_table(f'{Constants.data_dir}/{outcome}/{db}/{part}.parquet')
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

    return data


def get_cv_split(data, i=0, n_splits=5, seed=42):
    seeds = np.random.RandomState(seed).randint(low=0, high=2**32-1, size=(2, ))
    outer = KFold(n_splits, shuffle=True, random_state=seeds[0])
    inner = KFold(n_splits, shuffle=True, random_state=seeds[1])
    
    count = 0
    all_stays = data['sta'].index
    split = {"train": {}, "val": {}, "test": {}}
    for dev, test in outer.split(all_stays):
        for train, val in inner.split(dev):
            if count == i:
                split['train']['stays'] = all_stays[train]
                split['val']['stays'] = all_stays[val]
                split['test']['stays'] = all_stays[test]

                for s in split.keys():    # train / val / test 
                    for d in data.keys(): # sta / dyn / outc
                        split[s][d] = data[d].loc[split[s]['stays'], :]

                break
            count += 1
    
    return split


def preprocess_data(data, trial, n_splits, seed=None):
    # Encode sex into 0/1
    data['sta']['sex'] = (data['sta']['sex'] == "Female").astype(float)
    
    # Create train / val / test splits
    data = get_cv_split(data, trial, n_splits, seed)
    
    for part in ['sta', 'dyn']:
        # Calculate metrics from training split
        means = data['train'][part].mean(numeric_only=True)
        stds = data['train'][part].std()
        
        # Apply processsing to all folds using pre-calculated metrics where needed
        for fold in ['train', 'val', 'test']:
            df = data[fold][part]
            
            # Normalise
            df = (df - means) / stds

            # Add indicators for missingness
            inds = df.isna().astype(float)
            inds.columns = [name+"_ind" for name in inds.columns]
            df = pd.concat((df, inds), axis=1) 

            # Fill missing values
            df = df.groupby('stay_id').ffill()  # start with forward fill
            df = df.fillna(value=0)             # fill any remaining NAs with 0

            data[fold][part] = df

    return data


class Environment():
    def __init__(self, db, outcome, pad_to):
        self.db = db
        self.outcome = outcome
        self.pad_to = pad_to
                
    def prepare(self, trial, n_splits, seed=42, debug=False):
        data = load_data(self.db, self.outcome, debug)
        # Hack to avoid pandas/python unnecessarily hanging on to memory. Start
        # subprocess which is terminated afterwards, releasing all resources. 
        # See https://stackoverflow.com/questions/39100971/how-do-i-release-memory-used-by-a-pandas-dataframe 
        data = mp.Pool(1).apply(preprocess_data, args=[data, trial, n_splits, seed])
        self.data = data

    @property
    def loaded(self):
        return hasattr(self, 'data')

    def ascertain_loaded(self):
        if not self.loaded:
            raise RuntimeError(f'Data for {self.db}:{self.outcome} accessed before loading.')

    @property
    def num_inputs(self):
        self.ascertain_loaded()
        return len(self.data['train']['sta'].columns) + len(self.data['train']['dyn'].columns)

    def __getitem__(self, fold):
        self.ascertain_loaded()
        if not fold in ['train', 'val', 'test']:
            raise ValueError(f'fold must be one of ["train", "val", "test"], got {fold}')
        return Fold(self.data[fold], fold, self.pad_to)


class Fold(Dataset):
    def __init__(self, data, fold, pad_to=None):
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
            if len(Y.shape) > 1:
                Y = pad_to_len(Y, self.pad_to)   # pad_to x 1
                Y = Y[:, -1] 

        return X, Y

def pad_to_len(x, len):
    orig_len = x.shape[0]
    x_pad = np.full((len, x.shape[1]), PAD_VALUE, dtype=np.float32)
    x_pad[:orig_len, :] = x
    return x_pad
