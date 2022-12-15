from pathlib import Path
import argparse
import os
import torch
import torch.utils.data

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from statsmodels.gam.api import GLMGam, BSplines
import plotnine as ggp
from typing import List, Tuple

import torch

from icudg import tasks
from icudg import algorithms
from icudg.lib.fast_data_loader import FastDataLoader
from icudg.lib.misc import predict_on_set

from notebooks.utils import load_all_stats

def init_data_loader(envs: List[str], task: tasks.Task, fold: str) -> FastDataLoader:
    """Initialise a dataloader

    Args:
        env: name of one or more environments that should contribute to the dataloader 
        task: a set up task object containing the data
        fold: the resampling fold to load, can be 'train', 'val', or 'test'

    Returns:
        an initialised dataloader
    """
    return FastDataLoader(
        dataset=task.get_torch_dataset(envs, fold),  
        batch_size=1024,
        num_workers=1
    )

def get_predictions(algorithm: algorithms.Algorithm, loader: FastDataLoader) -> Tuple[np.ndarray, np.ndarray]:
    """Obtain predictions from a fitted algorithm 

    Args:
        algorithm: the fitted prediction model
        loader: the initialised data loader

    Returns:
        predictions and targets for all samples of the dataloader
    """
    preds, targets, _ = predict_on_set(algorithm, loader, 'cpu')
    preds = torch.softmax(preds, dim=-1)
    preds = preds[..., -1].flatten()
    targets = targets.flatten()
    mask = targets != 2
    return preds[mask].numpy(), targets[mask].numpy()

def enframe_result(
    raw: np.ndarray, 
    recal: np.ndarray, 
    targets: np.ndarray, 
    mode: str, 
    train: List[str], 
    test: str
) -> pd.DataFrame:
    """Combine raw and recalibrated prediction results in a pandas DataFrame

    Args:
        raw: raw model predictions
        recal: recalibrated model predictions
        targets: ground truth labels
        mode: fold for which the predictions were calculated, can be 'train', 'val', or 'test'
        train: names of all training environments
        test: name of the test environment

    Returns:
        all results in a single DataFrame
    """
    return pd.DataFrame({
        'mode': mode,
        'train': train[0] if len(train) == 1 else 'pooled',
        'test': test,
        'raw': raw, 
        'recal': recal, 
        'target': targets
    })

def smooth(preds: pd.Series, targets: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate smooth calibration curves using a Generalised Additive Model with BSplines

    Args:
        preds: model predictions
        targets: ground truth labels

    Note that the data is winsorised before fitting the GAM.

    Returns:
        smoothed values y over a grid x
    """
    q = preds.quantile([0.005, 0.995])
    keep = (preds > q.iloc[0]) & (preds < q.iloc[1])
    preds, targets = preds[keep], targets[keep]
    bs = BSplines(preds, df=4, degree=3)
    model = GLMGam(endog=targets, smoother=bs)
    fit = model.fit()
    x = np.arange(preds.min(), preds.max(), 0.01)
    y = fit.predict(exog_smooth=x)
    return x, y


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Domain calibration')
    parser.add_argument('--input-dir', type=str, default='/Users/patrick/clinicaldg-outputs/sepsis_best_tcn')
    parser.add_argument('--num-trials', type=int, default=10)
    args = parser.parse_known_args()[0]

    
    lst = load_all_stats(Path(args.input_dir))
    res = []

    for row_num, row in lst.iterrows():
        print(f"Process row {row_num+1} of {lst.shape[0]}.")
        train_args = row['args']
        train_hparams = row['model_hparams']
        
        algo = train_args['algorithm']
        if algo not in ["ERM", "ERMID"]:
            print(f"Skip as not ERM or ERMID.")
            continue
        elif train_args['trial'] != 1:
            print(f"Skip as not first fold.")
            continue
        
        test = row['model_test_domains']
        if algo == "ERM":
            train = list(set(['aumc', 'eicu', 'hirid', 'miiv']) - set(test))
        else: 
            train = test

        print(f"Load data.")
        task_class = vars(tasks)[train_args['task']]
        task = task_class(train_hparams, train_args)
        task.setup()

        print(f"Load model.")
        algorithm_class = vars(algorithms)[algo]
        algorithm = algorithm_class(task, None, train_hparams)
        algorithm.load_state_dict(
            torch.load(os.path.join(args.input_dir, test[0], row.folder, "model.pkl"))
        )

        print(f"Recalibrate.")
        train_loader = init_data_loader(train, task, 'train')
        train_preds, train_targets = get_predictions(algorithm, train_loader)
        val_loader = init_data_loader(train, task, 'val')
        val_preds, val_targets = get_predictions(algorithm, val_loader)

        iso = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip")
        iso.fit(train_preds, train_targets)

        train_recal = iso.predict(train_preds)
        val_recal = iso.predict(val_preds)

        res.append(enframe_result(train_preds, train_recal, train_targets, 'train', train, test[0]))
        res.append(enframe_result(val_preds, val_recal, val_targets, 'val', train, test[0]))

        if len(train) > 1:
            eval_on = test
        else:
            eval_on = ['aumc', 'eicu', 'hirid', 'miiv']
        
        print(f"Evaluate on: ", end="")
        for i, s in enumerate(eval_on):
            print(f"{s}", end="")
            test_loader = init_data_loader([s], task, 'test')
            test_preds, test_targets = get_predictions(algorithm, test_loader)
            test_recal = iso.predict(test_preds)
            res.append(enframe_result(test_preds, test_recal, test_targets, 'test', train, s))

            if i == len(eval_on):
                print("")
            else:
                print(", ", end="")

    res = pd.concat(res)

    res = res[(res['mode'] == 'test') & ((res['train'] == "pooled") | (res['train'] == res['test']))]

    res_smoothed = []
    for name, group in res.groupby(['mode', 'train', 'test']):
        grid, smoothed = smooth(group['recal'], group['target'])
        res_smoothed.append(pd.DataFrame({
            'mode': name[0], 
            'train': name[1], 
            'test': name[2],
            'x': grid, 
            'y': smoothed
        }))
    res_smoothed = pd.concat(res_smoothed)

    (ggp.ggplot(res_smoothed, ggp.aes('x', 'y')) 
     + ggp.geom_abline(intercept=0, slope=1, linetype="dotted", colour='lightgrey')
     + ggp.geom_line()
     + ggp.geom_rug(data=res[(res.train == res.test) & (res['mode'] == "test")], mapping=ggp.aes(x='recal', y=1), alpha=0.1)
     + ggp.coord_fixed(xlim=[0, 1], ylim=[-0.035, 1], expand=False)
     + ggp.facet_wrap('test')
     + ggp.theme_bw()
    )
