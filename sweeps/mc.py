import numpy as np
import pandas as pd
from itertools import product
from clinicaldg.lib import misc

# Define values to sweep over --------------------------------------------------
es_methods = ['train', 'val']
algorithms = ['ERMID', 'ERMMerged', 'ERM', 'IRM', 'VREx', 'RVP', 'CORAL',
              'IGA', 'MLDG', 'GroupDRO']
n_trials = 5
n_hparams = 30

# Generate grid ----------------------------------------------------------------
trial_seed = np.arange(n_trials)
hparams_seed = np.arange(n_hparams)

grid = product(es_methods, algorithms, trial_seed, hparams_seed)
grid = pd.DataFrame(grid, columns=['es', 'algorithm', 'trial_seed', 'hparams_seed'])

seeds = []
for i in range(grid.shape[0]):
    r = grid.iloc[i, :]
    s = misc.seed_hash("MultiCenter", r.algorithm, r.hparams_seed, r.trial_seed)
    seeds.append(s)
grid['seed'] = seeds

# Save to file -----------------------------------------------------------------
grid.to_csv(f'sweeps/mc_params.csv', index=False)