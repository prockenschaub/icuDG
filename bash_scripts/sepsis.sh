#!/bin/bash

#SBATCH --output=logs/%x-%A-%a.log
#SBATCH --ntasks=1
#SBATCH --mem=25G
#SBATCH --partition=medium
#SBATCH --time=36:00:00

cd ~/work/icuDG # NOTE: Change if your repo lives elsewhere

eval "$($(which conda) shell.bash hook)"
conda activate icudg

set -x

# If no architecture was set, default to TCN
if [ -z ${NN} ]
then 
  export NN=tcn
fi

while IFS="," read -r hs algo t v ts seed
do
    date 

    if test -f "outputs/sepsis_${NN}/${t}/run${SLURM_ARRAY_TASK_ID}/done"; then
      echo "Model already fitted (${t}/run${SLURM_ARRAY_TASK_ID}), do not run again."
      break
    fi

    python -m icudg.train \
        --task Sepsis \
        --algorithm ${algo} \
        --hparams "{\"test_env\": \"${t}\", \"val_env\": \"${v}\", \"architecture\": \"${NN}\"}" \
        --hparams_seed ${hs} \
        --trial ${ts} \
        --seed ${seed} \
        --es_metric val_nll \
        --es_patience 10 \
        --output_dir "outputs/sepsis_${NN}/${t}/run${SLURM_ARRAY_TASK_ID}" \
        --delete_model \
        --use_checkpoint

    date
done < <(sed -n "$((${SLURM_ARRAY_TASK_ID}+1))p" "sweeps/mc_params_train.csv")
