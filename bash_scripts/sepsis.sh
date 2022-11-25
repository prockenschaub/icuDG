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

# If no architecture was set, default to TCN
if [ -z ${NN} ]
then 
  export NN=tcn
fi

while IFS="," read -r hs algo t v ts seed
do
    date 

    python -m icudg.train \
        --task Sepsis \
        --algorithm ${algo} \
        --hparams "{\"test_env\": \"${t}\", \"val_env\": \"${v}\", \"mc_architecture\": \"${NN}\"}" \
        --hparams_seed ${hs} \
        --trial ${ts} \
        --seed ${seed} \
        --es_metric val_nll \
        --es_patience 20 \
        --checkpoint_freq 10 \
        --output_dir "outputs/sepsis_${NN}/${t}/run${SLURM_ARRAY_TASK_ID}"

    date
done < <(sed -n "$((${SLURM_ARRAY_TASK_ID}+1))p" "sweeps/mc_params_train.csv")
