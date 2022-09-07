#!/bin/bash

#SBATCH --output=logs/%x-%A-%a.log
#SBATCH --ntasks=2
#SBATCH --mem=50G
#SBATCH --partition=medium
#SBATCH --time=24:00:00

cd ~/work/ClinicalDG # NOTE: Change if your repo lives elsewhere

eval "$($(which conda) shell.bash hook)"
conda activate clinicaldg-new

set -x

if [ -z ${DS} ]
then 
  export DS=mimic
fi

while IFS="," read -r es algo ts hs seed
do
    date 

    python -m clinicaldg.scripts.train \
        --experiment MultiCenter \
        --algorithm ${algo} \
        --es_method ${es} \
        --hparams "{\"mc_target\": \"${DS}\", \"mc_architecture\": \"tcn\"}" \
        --hparams_seed ${hs} \
        --trial_seed ${ts} \
        --seed ${seed} \
        --output_dir "outputs/mc/${DS}/run${SLURM_ARRAY_TASK_ID}" \
        --delete_model

    date
done < <(sed -n "$((${SLURM_ARRAY_TASK_ID}+1))p" "sweeps/mc_params.csv")
