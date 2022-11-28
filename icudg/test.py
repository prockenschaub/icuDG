
import argparse
import pickle
import json
import os
import torch
import torch.utils.data

import pandas as pd
from sklearn.isotonic import IsotonicRegression
import plotnine as ggp

from icudg import tasks
from icudg import algorithms
from icudg.lib.fast_data_loader import FastDataLoader
from icudg.lib.misc import predict_on_set


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Domain generalization')
    parser.add_argument('--input_dir', type=str, default='/Users/patrick/clinicaldg-outputs/sepsis_best/miiv/run6')
    args = parser.parse_known_args()[0]

    with open(os.path.join(args.input_dir, "params.json"), 'r') as f:
        params = json.load(f)
        train_args = argparse.Namespace(**params['args'])
        train_hparams = params['hparams']

    with open(os.path.join(args.input_dir, "task.pkl"), 'rb') as f:
        task_info = pickle.load(f)
        

    task_class = vars(tasks)[train_args.task]
    task = task_class(train_hparams, vars(train_args))
    task.set_means_and_stds(task_info['means'], task_info['stds'])
    task.setup(envs=['miiv'], use_weight=False)
    task.set_weights(torch.tensor([1., 71], dtype=torch.float32))


    algorithm_class = vars(algorithms)[train_args.algorithm]
    algorithm = algorithm_class(task, None, train_hparams)
    algorithm.load_state_dict(
        torch.load(os.path.join(args.input_dir, "model.pkl"))
    )

    # Discrimintation
    test_loader = FastDataLoader(
        dataset=task.get_torch_dataset(['miiv'], 'val'),  
        batch_size=train_hparams['batch_size']*4,
        num_workers=train_args.num_workers
    )
    task.eval_metrics(algorithm, test_loader, device='cpu')    
    
    # Calibration
    def postprocess_preds(preds, targets):
        preds = torch.softmax(preds, dim=-1)
        preds = preds[..., -1].flatten()
        targets = targets.flatten()
        mask = targets != 2
        return preds[mask].numpy(), targets[mask].numpy()
    
    train_loader = FastDataLoader(
        dataset=task.get_torch_dataset(['miiv'], 'train'),  
        batch_size=train_hparams['batch_size']*4,
        num_workers=train_args.num_workers
    )
    train_preds, train_targets, _ = predict_on_set(algorithm, train_loader, 'cpu')
    train_preds, train_targets = postprocess_preds(train_preds, train_targets)
    iso = IsotonicRegression(y_min=0, y_max=1)
    iso.fit(train_preds, train_targets)


    val_loader = FastDataLoader(
        dataset=task.get_torch_dataset(['miiv'], 'val'),  
        batch_size=train_hparams['batch_size']*4,
        num_workers=train_args.num_workers
    )
    val_preds, val_targets, _ = predict_on_set(algorithm, val_loader, 'cpu')
    val_preds, val_targets = postprocess_preds(val_preds, val_targets)
    raw = pd.DataFrame({'preds': val_preds, 'targets': val_targets})
    (ggp.ggplot(raw, ggp.aes('preds', 'targets')) 
     + ggp.geom_smooth()
     + ggp.geom_abline(intercept=0, slope=1, linetype="dashed")
     + ggp.coord_fixed(xlim=[0, 1], ylim=[0, 1])
    )

    iso_preds = iso.predict(val_preds)
    recal = pd.DataFrame({'preds': iso_preds, 'targets': val_targets})
    (ggp.ggplot(recal, ggp.aes('preds', 'targets')) 
     + ggp.geom_smooth()
     + ggp.geom_abline(intercept=0, slope=1, linetype="dashed")
     + ggp.coord_fixed(xlim=[0, 1], ylim=[0, 1])
    )