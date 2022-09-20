import numpy as np
import pandas as pd
from itertools import product
from clinicaldg.lib import misc

# Define values to sweep over --------------------------------------------------
oracles = ['ERMID', 'ERMMerged']
algorithms = ['ERM', 'CORAL', 'VREx', 'IGA']
n_trials = 5
n_hparams = 10
envs = ['mimic', 'eicu', 'hirid', 'aumc']
trial_seed = np.arange(n_trials)
hparams_seed = np.arange(n_hparams)

# Generate grid ----------------------------------------------------------------

col_names = ['algorithm', 'test_env', 'val_env', 'trial_seed', 'hparams_seed']
oracle_grid = product(oracles, envs, ['train'], trial_seed, hparams_seed)
dg_grid = product(algorithms, envs, ['train'] + envs, trial_seed, hparams_seed)

grid = pd.concat((
    pd.DataFrame(oracle_grid, columns=col_names),
    pd.DataFrame(dg_grid, columns=col_names)
))

seeds = []
for i in range(grid.shape[0]):
    r = grid.iloc[i, :]
    s = misc.seed_hash("MultiCenter", r.algorithm, r.hparams_seed, r.trial_seed)
    seeds.append(s)
grid['seed'] = seeds

# Save to file -----------------------------------------------------------------
grid.to_csv(f'sweeps/mc_params.csv', index=False)