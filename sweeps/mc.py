import numpy as np
import pandas as pd
import argparse
from itertools import product
from icudg.lib import misc

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate hyperparam grid')
    parser.add_argument('--val', type=str, default="train", choices=['train', 'loo'])
    args = parser.parse_known_args()[0]

    # Define values to sweep over --------------------------------------------------
    algorithms = ['CORAL', 'VREx', 'Fishr', 'GroupDRO', 'MLDG']
    n_trials = 5
    n_hparams = 30
    envs = ['miiv', 'eicu', 'hirid', 'aumc']

    # Generate grid ----------------------------------------------------------------
    if args.val == 'train':
        val_envs = [args.val]
    elif args.val == 'loo':
        val_envs = envs

    trials = np.arange(n_trials)
    hparams_seed = np.arange(n_hparams) + 1

    col_names = ['hparams_seed', 'algorithm', 'test_env', 'val_env', 'trial']
    merge_grid = product(hparams_seed, ['ERMMerged'], ['all'], ['train'], trials)
    erm_grid = product(hparams_seed, ['ERMID', 'ERM'], envs, val_envs, trials)
    dg_grid = product(hparams_seed, algorithms, envs, val_envs, trials)

    grid = pd.concat((
        pd.DataFrame(merge_grid, columns=col_names),
        pd.DataFrame(erm_grid, columns=col_names)
    ))
    grid.sort_values('hparams_seed', kind='stable')
    grid = pd.concat((grid, pd.DataFrame(dg_grid, columns=col_names)))

    if args.val == 'loo':
        # Leave-on-dataset-out is not defined for oracle runs
        grid = grid[~grid['algorithm'].isin(['ERMMerged', 'ERMID'])]
        # Validation env must differ from test env in leave-on-dataset-out
        grid = grid[grid['val_env'] != grid['test_env']]

    seeds = []
    for i in range(grid.shape[0]):
        r = grid.iloc[i, :]
        s = misc.seed_hash("MultiCenter", r.algorithm, r.hparams_seed, r.trial)
        seeds.append(s)
    grid['seed'] = seeds

    # Save to file -----------------------------------------------------------------
    grid.to_csv(f'sweeps/mc_params_{args.val}.csv', index=False)