# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import argparse
import collections
import json
import os
import random
import sys
import time

import numpy as np
import PIL
import torch
import torchvision
import torch.utils.data

from clinicaldg import experiments
from clinicaldg import algorithms
from clinicaldg.lib import misc
from clinicaldg.lib.hparams_registry import HparamRegistry
from clinicaldg.lib.fast_data_loader import InfiniteDataLoader, FastDataLoader
from clinicaldg.lib.early_stopping import EarlyStopping
from clinicaldg.lib.checkpoint import has_checkpoint, load_checkpoint, save_checkpoint

torch.multiprocessing.set_sharing_strategy('file_system')

if __name__ == "__main__":
    # Define command line arguments
    parser = argparse.ArgumentParser(description='Domain generalization')
    parser.add_argument('--experiment', type=str, default="ColoredMNIST")
    parser.add_argument('--algorithm', type=str, default="ERM")
    parser.add_argument('--es_method', choices = ['train', 'val', 'test'])
    parser.add_argument('--hparams', type=str,
        help='JSON-serialized hparams dict')
    parser.add_argument('--hparams_seed', type=int, default=0,
        help='Seed for random hparams (0 means "default hparams")')
    parser.add_argument('--trial_seed', type=int, default=0,
        help='Trial number (used for seeding random_hparams).')
    parser.add_argument('--seed', type=int, default=0,
        help='Seed for everything else')
    parser.add_argument('--max_steps', type=int, default=None,
        help='Number of steps. Default is dataset-dependent.')
    parser.add_argument('--checkpoint_freq', type=int, default=None,
        help='Checkpoint every N steps. Default is dataset-dependent.')
    parser.add_argument('--output_dir', type=str, default="train_output")
    parser.add_argument('--delete_model', action = 'store_true', 
        help = 'delete model weights after training to save disk space')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    sys.stdout = misc.Tee(os.path.join(args.output_dir, 'out.txt'))
    sys.stderr = misc.Tee(os.path.join(args.output_dir, 'err.txt'))

    print("Environment:")
    print("\tPython: {}".format(sys.version.split(" ")[0]))
    print("\tPyTorch: {}".format(torch.__version__))
    print("\tTorchvision: {}".format(torchvision.__version__))
    print("\tCUDA: {}".format(torch.version.cuda))
    print("\tCUDNN: {}".format(torch.backends.cudnn.version()))
    print("\tNumPy: {}".format(np.__version__))
    print("\tPIL: {}".format(PIL.__version__))

    print('Args:')
    for k, v in sorted(vars(args).items()):
        print('\t{}: {}'.format(k, v))

    # Load the selected algorithm and experiment classes
    algorithm_class = vars(algorithms)[args.algorithm]
    experiment_class = vars(experiments)[args.experiment]     

    # Choose hyperparameters based on algorithm and experiment
    hparam_registry = HparamRegistry()
    hparam_registry.register(algorithm_class.HPARAM_SPEC)
    hparam_registry.register(experiment_class.HPARAM_SPEC)

    if args.hparams_seed == 0:
        hparams = hparam_registry.get_defaults()
    else:
        hparams = hparam_registry.get_random_instance(
            misc.seed_hash(args.hparams_seed, args.trial_seed)
        )
    if args.hparams:
        hparams.update(json.loads(args.hparams))

    print('HParams:')
    for k, v in sorted(hparams.items()):
        print('\t{}: {}'.format(k, v))
    

    # Seed everything
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Choose device (CPU or GPU)
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"               
    
    # Instantiate experiment
    experiment = experiment_class(hparams, args)

    # Confirm environment assignment (envs differ for Oracle runs)
    if args.algorithm == 'ERMID': # ERM trained on the training subset of the test env
        TRAIN_ENVS = experiment.TEST_ENVS
        VAL_ENVS = experiment.TEST_ENVS
        TEST_ENVS = experiment.TEST_ENVS
    elif args.algorithm == 'ERMMerged': # ERM trained on merged training subsets of all envs
        TRAIN_ENVS = experiment.ENVIRONMENTS
        VAL_ENVS = experiment.TEST_ENVS 
        TEST_ENVS = experiment.TEST_ENVS
    else:
        TRAIN_ENVS = experiment.TRAIN_ENVS
        VAL_ENVS = experiment.VAL_ENVS
        TEST_ENVS = experiment.TEST_ENVS
        
    print("Training Environments: " + str(TRAIN_ENVS))
    print("Validation Environments: " + str(VAL_ENVS))
    print("Test Environments: " + str(TEST_ENVS))    
  
    # Instantiate experiment and algorithm
    algorithm = algorithm_class(experiment, len(TRAIN_ENVS), hparams).to(device)

    # Get the datasets for each environment and split them into train/val/test
    train_dss = [experiment.get_torch_dataset([env], 'train') for env in TRAIN_ENVS]
    
    train_loaders = [
        InfiniteDataLoader(
            dataset=i,
            weights=None,
            batch_size=hparams['batch_size'],
            num_workers=experiment.N_WORKERS
        )
        for i in train_dss
        ]
    
    if args.es_method == 'train':
        val_ds = experiment.get_torch_dataset(TRAIN_ENVS, 'val')
    elif args.es_method == 'val':
        val_ds = experiment.get_torch_dataset(VAL_ENVS, 'val')
    elif args.es_method == 'test':
        val_ds = experiment.get_torch_dataset(TEST_ENVS, 'val')
        
    if hasattr(experiment, 'NUM_SAMPLES_VAL'):
        num_samples_val = min(experiment.NUM_SAMPLES_VAL, len(val_ds))
        val_idx = np.random.choice(np.arange(len(val_ds)), num_samples_val, replace = False)
        val_ds = torch.utils.data.Subset(val_ds, val_idx)

    val_loader = FastDataLoader(
        dataset=val_ds,
        batch_size=hparams['batch_size']*4,
        num_workers=experiment.N_WORKERS
    )
    
    test_loaders = {env:
        FastDataLoader(
            dataset=experiment.get_torch_dataset([env], 'test'),
            batch_size=hparams['batch_size']*4,
            num_workers=experiment.N_WORKERS
        )
        for env in experiment.ENVIRONMENTS   
    }
    
    print("Number of parameters: %s" % sum([np.prod(p.size()) for p in algorithm.parameters()]))

    # Load any existing checkpoints
    if has_checkpoint():
        state = load_checkpoint()
        algorithm.load_state_dict(state['model_dict'])
        
        if isinstance(algorithm.optimizer, dict):
            for k, opt in algorithm.optimizer.items():
                opt.load_state_dict(state['optimizer_dict'][k])
        else:
            algorithm.optimizer.load_state_dict(state['optimizer_dict'])
        
        [train_loader.sampler.load_state_dict(state['sampler_dicts'][c]) for c, train_loader in enumerate(train_loaders)]
        start_step = state['start_step']
        es = state['es']
        torch.random.set_rng_state(state['rng'])
        print("Loaded checkpoint at step %s" % start_step)
    else:
        start_step = 0    

    # Set any remaining training settings
    train_minibatches_iterator = zip(*train_loaders)   
    checkpoint_vals = collections.defaultdict(lambda: [])

    steps_per_epoch = min([len(i)/hparams['batch_size'] for i in train_dss])

    n_steps = args.max_steps or experiment.MAX_STEPS
    checkpoint_freq = args.checkpoint_freq or experiment.CHECKPOINT_FREQ
    
    es = EarlyStopping(patience = experiment.ES_PATIENCE)    
    last_results_keys = None


    # Main training loop -------------------------------------------------------
    for step in range(start_step, n_steps):
        # Check early stopping
        if es.early_stop:
            break

        # Forward pass and parameter update
        step_start_time = time.time()
        minibatches_device = [(misc.to_device(xy[0], device), misc.to_device(xy[1], device))
            for xy in next(train_minibatches_iterator)]
        algorithm.train()
        step_vals = algorithm.update(minibatches_device, device)
        checkpoint_vals['step_time'].append(time.time() - step_start_time)

        for key, val in step_vals.items():
            checkpoint_vals[key].append(val)

        # Validation and checkpointing
        if step % checkpoint_freq == 0:
            results = {
                'step': step,
                'epoch': step / steps_per_epoch,
            }

            for key, val in checkpoint_vals.items():
                results[key] = np.mean(val)

            val_metrics = experiment.eval_metrics(algorithm, val_loader, device=device)
            val_metrics = misc.add_prefix(val_metrics, "es")
            results.update(val_metrics)                        
                
            results_keys = sorted(results.keys())
            if results_keys != last_results_keys:
                misc.print_row(results_keys, colwidth=12)
                last_results_keys = results_keys
            misc.print_row([results[key] for key in results_keys],
                colwidth=12)

            results.update({
                'hparams': hparams,
                'args': vars(args)    
            })

            epochs_path = os.path.join(args.output_dir, 'results.jsonl')
            with open(epochs_path, 'a') as f:
                f.write(json.dumps(results, sort_keys=True) + "\n")
            
            save_checkpoint(algorithm, algorithm.optimizer, 
                            [train_loader.sampler.state_dict(train_loader._infinite_iterator) for c, train_loader in enumerate(train_loaders)], 
                            step+1, es, torch.random.get_rng_state())
            
            checkpoint_vals = collections.defaultdict(lambda: [])
            
            es(-results['es_' + experiment.ES_METRIC], step, algorithm.state_dict(), os.path.join(args.output_dir, "model.pkl"))            


    # Testing ------------------------------------------------------------------
    algorithm.load_state_dict(torch.load(os.path.join(args.output_dir, "model.pkl")))
    algorithm.eval()
    
    save_dict = {
        "args": vars(args),
        "model_input_shape": experiment.input_shape,
        "model_num_classes": experiment.num_classes,
        "model_train_domains": TRAIN_ENVS,
        "model_val_domain": VAL_ENVS,
        "model_test_domains": TEST_ENVS,
        "model_hparams": hparams,
        "es_step": es.step,
        'es_' + experiment.ES_METRIC: es.best_score
    }
    
    final_results = {}         
    for name, loader in test_loaders.items():
        test_metrics = experiment.eval_metrics(algorithm, loader, device=device)
        test_metrics = misc.add_prefix(test_metrics, name)
        final_results.update(test_metrics)
        
    save_dict['test_results'] = final_results    
        
    torch.save(save_dict, os.path.join(args.output_dir, "stats.pkl"))    

    with open(os.path.join(args.output_dir, 'done'), 'w') as f:
        f.write('done')

    if args.delete_model:
        os.remove(os.path.join(args.output_dir, "model.pkl"))
