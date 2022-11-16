import numpy as np
import pandas as pd
from typing import Dict
from sklearn.model_selection import KFold


def get_single_train_test_split(data: Dict[str, pd.DataFrame], train_pct: float = 0.7, val_pct: float = 0.1, seed: int = 42):
    """Randomly split the data into training, validation and test set

    Args:
        data:
        train_pct: . Defaults to 0.7.
        val_pct: . Defaults to 0.1.
        seed: . Defaults to 42.

    Returns:
        
    """
    stays = data['sta'].index.values
    stays = np.random.RandomState(seed).permutation(stays)

    num_stays = len(stays)
    delims = (num_stays * np.array([0, train_pct, train_pct + val_pct, 1])).astype(int)

    splits = {"train": {}, "val": {}, "test": {}}
    for i, fold in enumerate(splits.keys()):
        # Loop through train / val / test
        stays_in_fold = stays[delims[i] : delims[i + 1]]
        for type in data.keys():
            # Loop through dyn / sta / outc
            splits[fold][type] = data[type].loc[stays_in_fold, :]

    return splits


def get_cv_split(data: Dict[str, pd.DataFrame], i: int = 0, n_splits: int = 5, seed: int = 42):
    """Randomly split the data into training, validation and test set according to a nested CV scheme.

    Note: this function returns a single split from a repeated cross-validation.
          In order to get the full cross-validation, run with different values 
          of `i`. For example, running this function 10-times with values of 
          ranging from 0 to 9 will give 10 splits corresponding to a 2-times 
          repeated 5-fold cross-validation. 

    Args:
        data: 
        i: . Defaults to 0.
        n_splits: . Defaults to 5.
        seed: . Defaults to 42.

    Returns:
        
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
