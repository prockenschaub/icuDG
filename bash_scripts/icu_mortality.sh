#!/bin/bash

#SBATCH --output=logs/%x-%A-%a.log
#SBATCH --ntasks=2
#SBATCH --mem=15G
#SBATCH --partition=medium
#SBATCH --time=24:00:00

cd ~/work/ClinicalDG # NOTE: Change if your repo lives elsewhere

eval "$($(which conda) shell.bash hook)"
conda activate clinicaldg-new

set -x

while IFS="," read -r hs algo t v ts seed
do
    date 

    python -m icudg.train \
        --task Mortality24 \
        --algorithm ${algo} \
        --hparams "{\"test_env\": \"${t}\", \"val_env\": \"${v}\", \"architecture\": \"tcn\"}" \
        --hparams_seed ${hs} \
        --trial ${ts} \
        --seed ${seed} \
        --es_metric val_nll \
        --es_patience 20 \
        --checkpoint_freq 10 \
        --output_dir "outputs/icu-mortality/${t}/run${SLURM_ARRAY_TASK_ID}"

    date
done < <(sed -n "$((${SLURM_ARRAY_TASK_ID}+1))p" "sweeps/mc_params_train.csv")
