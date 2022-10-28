import numpy as np
import pandas as pd
from itertools import product
from clinicaldg.lib import misc

# Define values to sweep over --------------------------------------------------
algorithms = ['ERM', 'CORAL', 'VREx', 'Fishr', 'GroupDRO', 'MLDG']
n_trials = 5
n_hparams = 10
envs = ['miiv', 'eicu', 'hirid', 'aumc']
trials = np.arange(n_trials)
hparams_seed = np.arange(n_hparams) + 1

# Generate grid ----------------------------------------------------------------

col_names = ['algorithm', 'test_env', 'val_env', 'trial', 'hparams_seed']
merge_grid = product(['ERMMerged'], ['all'], ['train'], trials, hparams_seed)
id_grid = product(['ERMID'], envs, ['train'], trials, hparams_seed)
dg_grid = product(algorithms, envs, ['train'], trials, hparams_seed)

grid = pd.concat((
    pd.DataFrame(merge_grid, columns=col_names),
    pd.DataFrame(id_grid, columns=col_names),
    pd.DataFrame(dg_grid, columns=col_names)
))

seeds = []
for i in range(grid.shape[0]):
    r = grid.iloc[i, :]
    s = misc.seed_hash("MultiCenter", r.algorithm, r.hparams_seed, r.trial)
    seeds.append(s)
grid['seed'] = seeds

# Save to file -----------------------------------------------------------------
grid.to_csv(f'sweeps/mc_params.csv', index=False)