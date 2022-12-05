import plotnine as ggp
from scipy.stats import wilcoxon, ttest_rel

from utils import *
from icudg.lib import misc


task = 'aki'
path = Path(f'/Users/patrick/clinicaldg-outputs/{task}')

dbs = {
    'aumc': 'AUMCdb',
    'hirid': 'HiRID',
    'eicu': 'eICU',
    'miiv': 'MIMIC',
    'pooled (n-1)': 'pooled (n-1)',
    'all': 'all'
}

algos = ['ERM', 'CORAL', 'VREx', 'Fishr', 'MLDG', 'GroupDRO']

# ------------------------------------------------------------------------------
# Load and aggregate the results
grps = ['algorithm', 'val_env', 'test_env']
res = load_model_performance(path)
agg_res = aggregate_results(res, grps)
bst_res = pick_best_result(agg_res, grps)
summ = summ_mean_ste(bst_res, "auroc")

# ------------------------------------------------------------------------------
# Tabulate ERM performance
tbl = summ.melt(id_vars=grps+['hparams_seed'])

tbl['train_env'] = tbl['test_env']
tbl['test_env'] = tbl['variable'].str.extract('^([^_]*)(?=_)')

tbl = tbl[tbl['algorithm'].str.contains("ERM")]
tbl = tbl[tbl['variable'].str.contains("auroc")]
tbl = tbl[(tbl['algorithm'] != 'ERM') | (tbl.train_env == tbl.test_env)]

tbl.loc[tbl['algorithm'] == 'ERM', 'train_env'] = 'pooled (n-1)'

erm = tbl.pivot_table(
    values='value', 
    index='train_env', 
    columns = 'test_env', 
    aggfunc = lambda x: x
)
erm = erm.loc[list(dbs.keys()), list(dbs.keys())[:4]]
erm

erm.index = pd.Index(list(dbs.values()))
erm.to_csv(f'tables/erm_{task}.csv')

# ------------------------------------------------------------------------------
# Store grid for reruns with more trials
grid = tbl[['hparams_seed', 'algorithm', 'test_env', 'val_env', 'train_env']].copy()
grid = grid[(grid.algorithm != "ERMID") | (grid.train_env == grid.test_env)]
grid = grid.reset_index(drop=True)
grid.loc[:, 'trial'] = pd.Series([[i for i in range(10)] for _ in range(grid.shape[0])])
grid = grid.explode('trial')
grid.loc[grid['algorithm'] == "ERMMerged", 'test_env'] = "all"
grid = grid.drop_duplicates()
grid = grid.drop(columns='train_env')
grid['seed'] = grid.apply(lambda x: misc.seed_hash("MultiCenter", x.algorithm, x.hparams_seed, x.trial), axis=1)


grid.to_csv(f"sweeps/{task}_best.csv", index=False)

# ------------------------------------------------------------------------------
# Plot ERM performance
plt = tbl.copy()
plt['value_str'] = plt['value'].str.slice(stop=5)
plt['value'] = plt['value_str'].astype(float)

g = (ggp.ggplot(plt, ggp.aes('test_env', 'train_env'))
     + ggp.geom_tile(ggp.aes(fill='value'), colour='white', size=1)
     + ggp.geom_text(ggp.aes(label='value_str'), colour='white') 
     + ggp.geom_hline(yintercept=2.5, size=4, colour='white')
     + ggp.coord_equal() 
     + ggp.scale_x_discrete(
        limits=list(dbs.keys())[:4],
        labels=list(dbs.values())[:4]
       )
     + ggp.scale_y_discrete(
        limits=list(dbs.keys())[::-1],
        labels=list(dbs.values())[::-1]
       )
     + ggp.scale_fill_cmap(cmap_name='Blues', limits=[0.5, 1.0], expand=(0, 0))
     #+ ggp.guides(fill=ggp.guide_colourbar(barheight=50))
     + ggp.guides(fill=None)
     + ggp.labs(
        x='\nEvaluation dataset',
        y='Training dataset',
        fill=''
       )
     + ggp.theme_minimal()
     + ggp.theme(
        axis_text_x=ggp.element_text(size=12),
        axis_text_y=ggp.element_text(size=12, angle=45), 
        legend_text=ggp.element_text(size=12),
        panel_grid=ggp.element_blank()
       )
    )
g
ggp.ggsave(g, f'figures/erm_{task}.png', width=4, height=6, dpi=300)


# ------------------------------------------------------------------------------
# Tabulate DG performance
tbl = summ.melt(id_vars=grps+['hparams_seed'])

tbl['evaluated_in'] = tbl['variable'].str.extract('^([^_]*)(?=_)')

tbl = tbl[~tbl['algorithm'].isin(["ERMMerged", "ERMID"])]
tbl = tbl[tbl['test_env'] == tbl['evaluated_in']]

dg = tbl.pivot_table(
    values='value', 
    index='algorithm', 
    columns = 'evaluated_in', 
    aggfunc = lambda x: x
)
dg = dg.loc[algos, list(dbs.keys())[:4]]
dg
dg.to_csv(f'tables/dg_{task}.csv')

# ------------------------------------------------------------------------------
# Test for differences to ERM

exp = ['algorithm', 'test_env', 'hparams_seed']
setting = tbl[exp]
runs = setting.merge(res, on=exp)

drop_cols = [c for c in runs.columns if "es_" in c or "nll" in c]
perf = runs.drop(columns=drop_cols+['folder']).melt(id_vars=grps+['hparams_seed', 'trial'])
perf = perf[perf['variable'].str.extract("^([a-z]+)").values.squeeze() == perf['test_env'].values]

paired = perf[perf.algorithm == "ERM"].\
    drop(columns=['algorithm', 'val_env', 'hparams_seed', 'variable']).\
    merge(
        perf[perf.algorithm != "ERM"].drop(columns=['val_env', 'hparams_seed', 'variable']), 
        on=['test_env', 'trial'],
        suffixes=['_erm', '_dg']
    )

def compare_to_erm(df, test=ttest_rel):
    # Perform a one-sided test for hypothesis DG != ERM
    res = test(df.value_dg.to_numpy(), df.value_erm.to_numpy(), alternative="two-sided")
    return res.pvalue

paired.groupby(['algorithm', 'test_env']).apply(compare_to_erm)

# ------------------------------------------------------------------------------
# Plot DG training process

hparams = load_hparams(path)    

# TODO: allow for LOO
runs_hp = setting.merge(res[exp+['trial', 'folder', 'es_step']], on=exp).merge(hparams, on=exp)

db = 'miiv'
algo = 'GroupDRO'

progress = load_training_progress(path, db, runs_hp[(runs_hp['algorithm'] == algo) & (runs_hp['test_env'] == db)].folder)

if algo == "CORAL":
    plot_fun = plot_coral
elif algo == "VREx":
    plot_fun = plot_vrex
elif algo == "Fishr":
    plot_fun = plot_fishr
elif algo == "MLDG":
    plot_fun = plot_mldg
elif algo == "GroupDRO":
    plot_fun = plot_groupdro


plot_fun(runs.merge(progress, on='folder'))




# ------------------------------------------------------------------------------
# Plot DG hyperparameters

id_vars = ['algorithm', 'val_env', 'test_env', 'hparams_seed']
dg_params = hparams[~hparams['algorithm'].isin(["ERMMerged", "ERMID", "ERM"])].copy()
dg_params['chosen'] = False
dg_params.set_index(id_vars, inplace=True)
bst_idx = dg_params.index.intersection(bst_res.set_index(id_vars).index)
dg_params.loc[bst_idx, 'chosen'] = True
dg_params.reset_index(inplace=True)


plot_hparam(dg_params[dg_params['algorithm'] == "CORAL"], "mmd_gamma", [-1, 1])
plot_hparam(dg_params[dg_params['algorithm'] == "VREx"], "vrex_lambda", [-1, 5])
plot_hparam(dg_params[dg_params['algorithm'] == "Fishr"], "fishr_lambda", [1, 4])
plot_hparam(dg_params[dg_params['algorithm'] == "MLDG"], "mldg_beta", [-1, 1])
plot_hparam(dg_params[dg_params['algorithm'] == "GroupDRO"], "groupdro_eta", [-3, -1])

