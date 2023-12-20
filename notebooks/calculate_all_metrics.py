import argparse
import pickle
import json
import os
import torch
import torch.utils.data

from icudg import tasks
from icudg import algorithms
from icudg.lib.fast_data_loader import FastDataLoader


def calculate_metrics(model_dir):
    with open(os.path.join(model_dir, "params.json"), 'r') as f:
        params = json.load(f)
        train_args = params['args']
        train_hparams = params['hparams']

    with open(os.path.join(model_dir, "task.pkl"), 'rb') as f:
        task_info = pickle.load(f)

    task_class = vars(tasks)[train_args['task']]

    if train_hparams['test_env'] == "all":
        envs = task_class.ENVIRONMENTS
    else:
        envs = [train_hparams['test_env']] 

    task = task_class(train_hparams, train_args)
    task.set_means_and_stds(task_info['means'], task_info['stds'])
    task.setup(envs=envs, use_weight=False)
    task.set_weights(torch.tensor([1., 71], dtype=torch.float32))


    algorithm_class = vars(algorithms)[train_args['algorithm']]
    algorithm = algorithm_class(task, None, train_hparams)
    algorithm.load_state_dict(
        torch.load(os.path.join(model_dir, "model.pkl"))
    )

    # Discrimination
    res = {}
    for env in envs:
        test_loader = FastDataLoader(
            dataset=task.get_torch_dataset([env], 'test'),  
            batch_size=train_hparams['batch_size']*4,
            num_workers=train_args['num_workers']
        )
        res[env] = task.eval_metrics(algorithm, test_loader, device='cpu')  

    with open(os.path.join(model_dir, "all_metrics.pkl"), 'w') as f:
        json.dump(res, f)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Domain generalization')
    parser.add_argument('--input_dir', type=str, default='/Users/patrick/icudg-outputs/icu-mortality_gru_refit')
    args = parser.parse_known_args()[0]

    print(f"Calculate metrics for all models in {args.input_dir}\n")
    dss = os.listdir(args.input_dir)
    dss.sort()
    for ds in dss:
        print(f"   Dataset {ds}")
        runs = os.listdir(os.path.join(args.input_dir, ds))
        runs.sort()
        for run in runs:
            print(f"      {run}")
            calculate_metrics(os.path.join(args.input_dir, ds, run))

    
        
    


