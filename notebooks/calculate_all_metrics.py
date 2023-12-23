import argparse
import pickle
import json
import os
import torch
import torch.utils.data

from pathlib import Path

import numpy as np

from icudg import tasks
from icudg import algorithms
from icudg.lib.fast_data_loader import FastDataLoader

from icudg.lib.misc import predict_on_set, cat
from icudg.lib.metrics import roc_auc_score, roc_curve, average_precision_score, precision_recall_curve, recall_score, precision_score

def calculate_metrics(model_dir):
    with open(os.path.join(model_dir, "params.json"), 'r') as f:
        params = json.load(f)
        train_args = params['args']
        train_hparams = params['hparams']

    with open(os.path.join(model_dir, "task.pkl"), 'rb') as f:
        task_info = pickle.load(f)

    task_class = vars(tasks)[train_args['task']]

    if train_args['algorithm'] == 'ERMMerged':
        trn_envs = task_class.ENVIRONMENTS
        tst_envs = task_class.ENVIRONMENTS
    elif train_args['algorithm'] == 'ERMID':
        trn_envs = [model_dir.parent.name]
        tst_envs = task_class.ENVIRONMENTS
    elif train_args['algorithm'] == 'ERM':
        trn_envs = [env for env in task_class.ENVIRONMENTS if env != model_dir.parent.name]
        tst_envs = [train_hparams['test_env']] 

    task = task_class(train_hparams, train_args)
    task.set_means_and_stds(task_info['means'], task_info['stds'])
    task.setup(envs=[env for env in task_class.ENVIRONMENTS if env in trn_envs or env in tst_envs], use_weight=False)
    task.set_weights(torch.tensor([1., 71], dtype=torch.float32))


    algorithm_class = vars(algorithms)[train_args['algorithm']]
    algorithm = algorithm_class(task, None, train_hparams)
    algorithm.load_state_dict(
        torch.load(os.path.join(model_dir, "model.pkl"))
    )

    # Discrimination
    res = {}
    
    val_loader = FastDataLoader(
            dataset=task.get_torch_dataset(trn_envs, 'val'),  
            batch_size=train_hparams['batch_size']*4,
            num_workers=train_args['num_workers']
        )
    
    for env in tst_envs:
        tst_loader = FastDataLoader(
            dataset=task.get_torch_dataset([env], 'test'),  
            batch_size=train_hparams['batch_size']*4,
            num_workers=train_args['num_workers']
        )
        res[env] = do_calc(task, algorithm, val_loader, tst_loader, device='cpu')  
        #res[env] = task.eval_metrics(algorithm, tst_loader, device='cpu')  

    with open(os.path.join(model_dir, "all_metrics.pkl"), 'w') as f:
        json.dump(res, f)


def get_logits_and_outcome(task, algorithm, loader, device='cpu'):
    logits, y, mask = predict_on_set(algorithm, loader, device, task.get_mask)
    mask = cat(mask).to(device)

    probs = torch.sigmoid(logits[..., -1])
    
    probs = probs.cpu().numpy()
    y = y.long().cpu().numpy()
    mask = mask.int().cpu().numpy()

    probs_max, y_max = probs, y

    if len(y.shape) == 2:
        y_max = (y * mask).max(axis=-1)
        probs_max = (probs * mask).max(axis=-1)

    probs = probs.reshape(-1)[mask.reshape(-1) == 1]
    y = y.reshape(-1)[mask.reshape(-1) == 1]

    return probs, y, probs_max, y_max

def do_calc(task, algorithm, val_loader, tst_loader, device='cpu'):
        _, _, val_p_max, val_y_max = get_logits_and_outcome(task, algorithm, val_loader, device)
        tst_p, tst_y, tst_p_max, tst_y_max = get_logits_and_outcome(task, algorithm, tst_loader, device)
        
        auroc = roc_auc_score(tst_y, tst_p)
        auprc = average_precision_score(tst_y, tst_p)

        # Get sensitivity and PPV at the level of ICU admissions
        def calc_ppv_at(sens):
            _, roc_sens, roc_thresh = roc_curve(val_y_max, val_p_max)
            candidates = roc_thresh[roc_sens >= sens]
            if len(candidates) > 0:
                sens_thresh = max(candidates)
                return precision_score(tst_y_max, tst_p_max >= sens_thresh)
            else:
                return np.nan

        ppvs = {sens: calc_ppv_at(sens) for sens in np.arange(0.05, 0.95, 0.05)}

        def calc_sens_at(ppv):
            prc_ppv, _, prc_thresh = precision_recall_curve(val_y_max, val_p_max)
            candidates = prc_thresh[prc_ppv[:-1] >= ppv]
            if len(candidates) > 0:
                ppv_thresh = min(candidates)
                return recall_score(tst_y_max, tst_p_max >= ppv_thresh)
            else:
                return np.nan

        senss = {ppv: calc_sens_at(ppv) for ppv in np.arange(0.05, 0.95, 0.05)}

        return {
            'auroc': auroc, 
            'auprc': auprc,
            'ppvs': ppvs,
            'senss': senss
        }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Domain generalization')
    parser.add_argument('--input_dir', type=str, default='/Users/patrick/icudg-outputs/aki_gru_refit-5')
    args = parser.parse_known_args()[0]

    input_dir = Path(args.input_dir)

    print(f"Calculate metrics for all models in {args.input_dir}\n")
    dss = [f for f in input_dir.iterdir() if f.is_dir()]
    dss.sort()
    for ds in dss:
        print(f"   Dataset {ds.name}")
        runs = [f for f in ds.iterdir() if f.is_dir()]
        runs.sort()
        for run in runs:
            print(f"      {run.name}")
            calculate_metrics(run)

    
        
    


