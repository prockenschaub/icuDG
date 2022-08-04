# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from torchvision.datasets import MNIST
from pathlib import Path

from clinicaldg.experiments.mnist import Constants as mnistConstants

# MNIST #######################################################################

def download_mnist():
    # Original URL: http://yann.lecun.com/exdb/mnist/
    Path(mnistConstants.mnist_dir).mkdir(exist_ok = True, parents = True)
    MNIST(mnistConstants.mnist_dir, download=True)

if __name__ == "__main__":
    download_mnist()