from pathlib import Path
import re
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from typing import List, Dict
import plotnine as ggp

METHODS = ['ERMMerged', 'ERMID', 'ERM',  'CORAL', 'VREx', 'Fishr', 'GroupDRO', 'MLDG']

def load_single_set_of_stats(path: Path, subset: List = None) -> Dict:
    try:
        stats = torch.load((path/'stats.pkl').open('rb'))
    except: 
        return None 
    stats['folder'] = path.stem
    
    if subset is None:
        return stats
    
    res = {}
    for elem in subset:
        x = stats
        for k in elem.split("/"):
            x = x[k]
        if isinstance(x, dict):
            res.update(x)
        else:
            res[k] = x
    return res

def load_all_stats(path: Path, subset: List = None) -> pd.DataFrame:
    lst = []
    for i in tqdm(path.glob('**/done')):
        stats = load_single_set_of_stats(i.parent, subset)
        if stats is not None:
            lst.append(stats)
    return pd.DataFrame(lst)

def load_model_performance(path: Path) -> List:
    """Load all 

    Args:
        path: _description_

    Returns:
        _description_
    """
    subset = [
        "args/algorithm",
        "model_hparams/val_env",
        "model_hparams/test_env",
        "args/hparams_seed",
        "args/trial",
        "folder",
        "es_step",
        "es_val_nll",
        "test_results"
    ]
    
    df = load_all_stats(path, subset)
    df['algorithm'] = pd.Categorical(df['algorithm'], categories=METHODS)
    df = df.sort_values(['algorithm', 'test_env', 'hparams_seed', 'trial'], ignore_index=True)
    return df

def load_hparams(path: Path) -> pd.DataFrame:
    subset = [
        "args/algorithm",
        "model_hparams/test_env",
        "args/hparams_seed",
        "model_hparams"
    ]
    df = load_all_stats(path, subset)
    df['algorithm'] = pd.Categorical(df['algorithm'], categories=METHODS)
    df = df.sort_values(['algorithm', 'test_env', 'hparams_seed'], ignore_index=True)
    return df.drop_duplicates()
    
def aggregate_results(df, by):
    return df.\
        drop(columns=['trial', 'folder', 'es_step']).\
        groupby(by+['hparams_seed'], observed=True).\
        agg(['count', 'mean', 'std']).\
        reset_index()

def pick_best_result(df, by):
    return df.\
        groupby(by, observed=True, as_index=False).\
        apply(lambda x: x.loc[x['es_val_nll']['mean'].idxmin(), :])

def summ_mean_ste(df, keep=None):
    df = df.copy()
    meta = [c for c, s in df.columns if s == ""]
    cols = [c for c, s in df.columns if s != ""]
    _, idx = np.unique(cols, return_index=True)
    cols = [cols[i] for i in sorted(idx)]
    df.columns = ["_".join(p) if p[1] != '' else p[0] for p in df.columns]
    if keep:
        cols = [c for c in cols if re.search(keep, c)]
    for c in cols:
        df[f'{c}_ste'] = df[f'{c}_std'] / df[f'{c}_count'] ** (1/2)
        df[c] = df[f'{c}_mean'].apply('{:.3f}'.format) + u"\u00B1" + df[f'{c}_ste'].apply('{:.3f}'.format)
    return df[meta+cols]


def load_training_progress(path, db, folders):
    progress = []
    for folder in folders:
        file = Path(path, db, folder, "results.jsonl")
        df = pd.read_json(file, lines=True)
        df['folder'] = folder
        progress.append(df)
    return pd.concat(progress, axis=0)


def plot_base(progress: pd.DataFrame) -> ggp.ggplot:
    return (
        ggp.ggplot(progress, ggp.aes('step', group='trial'))
            + ggp.geom_line(ggp.aes(y='loss', color=['loss']), show_legend=True) 
            + ggp.geom_line(ggp.aes(y='val_nll', color=['val']), show_legend=True)
            + ggp.geom_vline(ggp.aes(xintercept='es_step'), colour='black', show_legend=True)
            + ggp.scale_color_discrete(name='', limits=['loss', 'val', 'nll', 'penalty'])
            + ggp.facet_wrap("~ trial")
            + ggp.labs(x='Training step', y='Loss')
            + ggp.theme_bw()
    )

def plot_coral(progress: pd.DataFrame) -> ggp.ggplot:
    progress['penalty_scld'] = progress['penalty'] * progress['mmd_gamma']
    return (
        plot_base(progress)
        #    + ggp.geom_line(ggp.aes(y='nll', color=['nll']), show_legend=True) TODO: add back in when refactored code is run
            + ggp.geom_line(ggp.aes(y='penalty', color=['penalty']), linetype='dashed')
            + ggp.geom_line(ggp.aes(y='penalty_scld', color=['penalty']))
            
            + ggp.labs(title='CORAL training progress')
    )

def plot_vrex(progress: pd.DataFrame) -> ggp.ggplot:
    progress['penalty_scld'] = progress['penalty'] * progress['vrex_lambda']
    return (
        plot_base(progress)
            + ggp.geom_line(ggp.aes(y='nll', color=['nll']), show_legend=True) 
            + ggp.geom_line(ggp.aes(y='penalty', color=['penalty']), linetype='dashed')
            + ggp.geom_line(ggp.aes(y='penalty_scld', color=['penalty']))
            + ggp.geom_vline(ggp.aes(xintercept='vrex_penalty_anneal_iters'), colour='black', linetype='dashed', show_legend=True)
            + ggp.labs(title='VREx training progress')
    )

def plot_fishr(progress: pd.DataFrame) -> ggp.ggplot:
    progress['penalty_scld'] = progress['penalty'] * progress['fishr_lambda']
    return (
        plot_base(progress)
            + ggp.geom_line(ggp.aes(y='nll', color=['nll']), show_legend=True) 
            + ggp.geom_line(ggp.aes(y='penalty', color=['penalty']), linetype='dashed')
            + ggp.geom_line(ggp.aes(y='penalty_scld', color=['penalty']))
            + ggp.geom_vline(ggp.aes(xintercept='fishr_penalty_anneal_iters'), colour='black', linetype='dashed', show_legend=True)
            + ggp.labs(title='Fishr training progress')
    )

def plot_mldg(progress: pd.DataFrame) -> ggp.ggplot:
    return (
        plot_base(progress)
            + ggp.labs(title='MLDG training progress')
    )

def plot_groupdro(progress: pd.DataFrame) -> ggp.ggplot:
    return (
        plot_base(progress)
            + ggp.labs(title='GroupDRO training progress')
    )


def plot_hparam(hparams, param, limits):
    return (
        ggp.ggplot(hparams, ggp.aes('test_env', param, colour='chosen', shape='chosen'))
        + ggp.geom_point(position=ggp.position_jitter(width=0.2), size=5, show_legend=False)
        + ggp.scale_y_log10()
        + ggp.scale_x_discrete(
            limits=['aumc', 'hirid', 'eicu', 'miiv'],
            labels=['AUMC', 'HiRID', 'eICU', 'MIMIC']
        )
        + ggp.scale_colour_discrete(limits=[True, False])
        + ggp.scale_shape_manual(name=" ", values=["*", "."], limits=[True, False])
        + ggp.coord_cartesian(ylim=limits)
        + ggp.labs(x=" ")
        + ggp.theme_bw()
    )
