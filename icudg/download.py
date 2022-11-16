# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import subprocess
import os
import shutil
from pathlib import Path

import pandas as pd
from tqdm import tqdm
from torchvision.datasets import MNIST

from icudg.tasks.mnist import Constants as mnistConstants
from icudg.tasks.physionet import Constants as pnConstants

def safe_mkdir(path):
    if not path.exists():
        answer = input(f"Directory {path} does not exist. Should it be created (y/n)?")
        if answer != "y":
            print("Cancelled data download.")
            return 
        path.mkdir(parents=True, exist_ok=True)

# MNIST ------------------------------------------------------------------------

def download_mnist():
    # Original URL: http://yann.lecun.com/exdb/mnist/
    path = Path(mnistConstants.data_dir)
    safe_mkdir(path)
    MNIST(mnistConstants.data_dir, download=True)


# PhysioNet CinC Challenge 2019 ------------------------------------------------

def download_physionet2019(path):
    path = Path(path)
    safe_mkdir(path)
    
    # Download via wget
    url = "https://physionet.org/files/challenge-2019/1.0.0/"
    oldpwd = os.getcwd()
    os.chdir(path)
    try:
        subprocess.run(["wget", "-r", "-N", "-c", "-np", url])
    finally:
        os.chdir(oldpwd)
    
    # Move data files to root and delete unnecessary metadata
    path_to_actual_data = path/'physionet.org'/'files'/'challenge-2019'/'1.0.0'/'training'
    shutil.move(path_to_actual_data/'training_setA', path)
    shutil.move(path_to_actual_data/'training_setB', path)
    shutil.rmtree(path/'physionet.org')


def import_physionet2019():
    lst = []
    for db in ['training_setA', 'training_setB']:
        path = Path(pnConstants.data_dir, db)
        for i in tqdm(path.glob("**/p*.psv")):
            df = pd.read_table(i, delimiter="|")
            df.insert(0, "Id", int(i.stem[1:]))
            lst.append(df)
        df = pd.concat(lst, axis=0)
        df.to_parquet(path/f'{db}.parquet')


if __name__ == "__main__":
    download_mnist()
    download_physionet2019()
    import_physionet2019()