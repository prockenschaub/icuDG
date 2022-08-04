# Reproduced from the eICU Benchmarks repository
# https://github.com/mostafaalishahi/eICU_Benchmark
from __future__ import absolute_import
from __future__ import print_function

from . import utils

def data_extraction_mortality(root_dir, time_window = 48):
    all_df = utils.embedding(root_dir)
    all_mort = utils.filter_mortality_data(all_df)
    all_mort = all_mort[all_mort['itemoffset']<=time_window]
    return all_mort