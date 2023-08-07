# Generalisability of deep learning-based early warning in the intensive care unit

## Paper

If you use this code in your research, please cite the following publication:

```
@article{rockenschaub2023generalisability,
      title={From Single-Hospital to Multi-Centre Applications: Enhancing the Generalisability of Deep Learning Models for Adverse Event Prediction in the ICU}, 
      author={Patrick Rockenschaub and Adam Hilbert and Tabea Kossen and Falk von Dincklage and Vince Istvan Madai and Dietmar Frey},
      year={2023},
      eprint={2303.15354},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}

```

This paper can be found on arxiv: https://arxiv.org/abs/2303.15354

## Acknowledgements

This implementation is a modified version of the [ClinicalDG](https://github.com/MLforHealth/ClinicalDG) framework (from commit [72154a8](https://github.com/MLforHealth/ClinicalDG/tree/72154a87a6d36416c0dac36e7a846b1194c7f39c)), which in turn is based on the well-known [DomainBed](https://github.com/facebookresearch/DomainBed) framework (from commit [a10458a](https://github.com/facebookresearch/DomainBed/tree/a10458a2adfd8aec0fda2d617f710e5044e5dc60)). 


## To replicate the experiments in the paper:

### Step 0: Environment and Prerequisites

Run the following commands to clone this repo and create the Conda environment:

```
git clone https://github.com/prockenschaub/icuDG.git
cd icuDG/
conda env create -f environment.yml
conda activate icudg
```

All experiments were run using Python 3.9.12 on an Apple M1 Max with Ventura 13.2.1 and on a Linux HPC cluster. 

### Step 1: Obtaining the Data

The main experiment data was collected from four open-source ICU datasets: 

* [AUMCdb](https://github.com/AmsterdamUMC/AmsterdamUMCdb)
* [HiRID](https://hirid.intensivecare.ai/)
* [eICU](https://eicu-crd.mit.edu/)
* [MIMIC IV](https://mimic.mit.edu/)

Information on how to gain access to these datasets can be found at the respective links provided above. Prediction tasks were derived from the raw data using the [`ricu` R package](https://github.com/eth-mds/ricu). The corresponding code can be found in the [following accompanying repo](https://github.com/prockenschaub/icuDG-preprocessing).

#### Dummy data

In order to facilitate trial runs of the domain generalisation algorithms provided in this repo, exammple code was provided that models immediately available sepsis data from the CinC Physionet Challenge 2019. You can download the data using:

```
python -m icudg.tasks.physionet.download
```

Note that the openly available Physionet data only contains 2 environments (hospitals A and B), meaning that no independent environment can be put aside for testing. The data is therefore merely meant for trial runs and debugging purposes. 

#### Specifying file location

The file paths to the data need to be specified in [config.yml](config.yml). 

For the `MulticenterICU` task, the path should point to a folder with a subfolders per prediction type (`mortality24`, `aki`, `sepsis`). Each of those folders should then contain one folder per dataset: `aumc`, `hirid`, `eicu`, `miiv`, each containing the preprocessed data.

For the `PhysioNet2019` task, the path should point to the folder created by the download script.


### Step 2: Running Experiments

Experiments can be ran using a similar procedure as for the [DomainBed](https://github.com/facebookresearch/DomainBed) framework. 

For example, to train a single model: 

```
python -m icudg.train \
        --task PhysioNet2019 \
        --algorithm ERM \
        --hparams '{"val_env": "train", "test_env": "training_setA",  "architecture": "gru"}' \
        --es_metric val_nll \
        --es_patience 10 \
        --output_dir "outputs/physionet" 
```

A full list of the available command line arguments can be found using `python -m icudg.train --help`. Available hyperparameters for training differ by task and algorithm and can be found in the `HPARAM_SPEC` attribute of the respective classes. 

To perform random hyperparameter sweeps, a list of hyperparameters was drawn and stored in the [sweeps/](sweeps) subfolder. Based on these hyperparameter lists, sweeps were run on our HPC using SLURM's `sbatch --array` using the scripts in [bash_scripts/](bash_scripts). These files are tailored to our setup and will likely require adjustments to your setup.

### Step 3: Aggregating Results

We provide sample code for creating aggregate results for an experiment in [notebooks/agg_results.ipynb](notebooks/agg_results.ipynb).


## License
This source code is released under the MIT license, included [here](LICENSE).
