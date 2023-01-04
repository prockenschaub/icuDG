#!/bin/bash

#SBATCH --output=logs/%x-%A-%a.log
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --partition=medium
#SBATCH --time=48:00:00

cd ~/work/icuDG # NOTE: Change if your repo lives elsewhere

eval "$($(which conda) shell.bash hook)"
conda activate icudg

set -x

# If no START was set, default to 1
if [ -z ${START} ]
then 
  export START=1
fi


# If no architecture was set, default to TCN
if [ -z ${NN} ]
then 
  export NN=tcn
fi


for i in $(seq 50)
do
  row=$(( $START + $i - 1 ))

  while IFS="," read -r hs algo t v ts seed
  do
      date 

      python -m icudg.train \
        --task Mortality24 \
        --algorithm ${algo} \
        --hparams "{\"test_env\": \"${t}\", \"val_env\": \"${v}\", \"architecture\": \"${NN}\"}" \
        --hparams_seed ${hs} \
        --trial ${ts} \
        --seed ${seed} \
        --es_metric val_nll \
        --es_patience 10 \
        --output_dir "outputs/icu-mortality_${NN}/${t}/run${SLURM_ARRAY_TASK_ID}" \
        --delete_model & 

      date
  done < <(sed -n "$(($row+1))p" "sweeps/mc_params_train.csv")
done
