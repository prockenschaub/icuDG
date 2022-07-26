# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from torchvision.datasets import MNIST
import xml.etree.ElementTree as ET
from zipfile import ZipFile
import argparse
import tarfile
import shutil
import uuid
import json
import os
from pathlib import Path
import yaml

# MNIST #######################################################################

def download_mnist():
    # Find path where to store the downloaded MNIST data
    with open("config.yml", "r") as stream:
        config = yaml.safe_load(stream)
    mnist_dir = config['mnist']['mnist_dir']

    # Original URL: http://yann.lecun.com/exdb/mnist/
    Path(mnist_dir).mkdir(exist_ok = True, parents = True)
    MNIST(mnist_dir, download=True)

if __name__ == "__main__":
    download_mnist()