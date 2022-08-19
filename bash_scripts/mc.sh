#!/bin/bash

#SBATCH --job-name=cdg-mc
#SBATCH --output=logs/%x-%A-%a.log
#SBATCH --ntasks=2
#SBATCH --mem=50G
#SBATCH --partition=medium
#SBATCH --time=15:00:00

cd ~/work/ClinicalDG # NOTE: Change if your repo lives elsewhere

eval "$($(which conda) shell.bash hook)"
conda activate clinicaldg

set -x

while IFS="," read -r es algo ts hs seed
do
    date 

    python -m clinicaldg.scripts.train \
        --experiment MultiCenterMIMIC \
        --algorithm ${algo} \
        --es_method ${es} \
        --hparams '{"mc_architecture": "tcn"}' \
        --hparams_seed ${hs} \
        --trial_seed ${ts} \
        --seed ${seed} \
        --output_dir "outputs/mc/mimic/run${SLURM_ARRAY_TASK_ID}" \
        --delete_model
done < <(sed -n "$((${SLURM_ARRAY_TASK_ID}+1))p" "sweeps/mc_params.csv")