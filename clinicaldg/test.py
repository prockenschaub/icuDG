
import argparse
import json
import os
import torch
import torch.utils.data

from clinicaldg import tasks
from clinicaldg import algorithms
from clinicaldg.lib.fast_data_loader import FastDataLoader


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Domain generalization')
    parser.add_argument('--input_dir', type=str, default='/Users/patrick/clinicaldg-outputs/charite')
    args = parser.parse_known_args()[0]

    with open(os.path.join(args.input_dir, "params.json"), 'r') as f:
        params = json.load(f)
        train_args = argparse.Namespace(**params['args'])
        train_hparams = params['hparams']

    train_args.debug = False # FIXME: remove after this feature is included in the results

    task_class = vars(tasks)[train_args.task]
    task = task_class(train_hparams, train_args)
    task.TRAIN_PCT = 1. # label everything as train to normalisation based on entire dataset
    task.VAL_PCT = 0.
    task.setup(envs=['mimic'], use_weight=False)

    algorithm_class = vars(algorithms)[train_args.algorithm]
    algorithm = algorithm_class(task, None, train_hparams)
    algorithm.load_state_dict(
        torch.load(os.path.join(args.input_dir, "model.pkl"))
    )

    test_loader = FastDataLoader(
        dataset=task.get_torch_dataset(['mimic'], 'train'),  
        batch_size=train_hparams['batch_size']*4,
        num_workers=args.num_workers
    )
    task.eval_metrics(algorithm, test_loader, device='cpu')    