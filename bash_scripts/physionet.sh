#!/bin/bash

#SBATCH --output=logs/%x-%A-%a.log
#SBATCH --ntasks=2
#SBATCH --mem=25G
#SBATCH --partition=medium
#SBATCH --time=36:00:00

cd ~/work/ClinicalDG # NOTE: Change if your repo lives elsewhere

eval "$($(which conda) shell.bash hook)"
conda activate clinicaldg-new

set -x

while IFS="," read -r algo t v ts hs seed
do
    date 

    python -m clinicaldg.train \
        --task PhysioNet2019 \
        --algorithm ${algo} \
        --hparams "{\"test_env\": \"${t}\", \"val_env\": \"${v}\", \"architecture\": \"tcn\"}" \
        --hparams_seed ${hs} \
        --trial ${ts} \
        --seed ${seed} \
        --es_metric val_nll \
        --es_patience 10 \
        --output_dir "outputs/physionet/${t}/run${SLURM_ARRAY_TASK_ID}" \
        --delete_model

    date
done < <(sed -n "$((${SLURM_ARRAY_TASK_ID}+1))p" "sweeps/pn19_params.csv")
