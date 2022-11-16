# Based on code by Zhang et al., rearranged and refactored by 
# Patrick Rockenschaub. 

import numpy as np

import torch
from torchvision.datasets import MNIST

from . import Constants 


class ColoredMNISTDataset():
    def __init__(self, hparams, args):        
        mnist = MNIST(Constants.data_dir, train=True, download=True)
        mnist_train = [mnist.data[:50000], mnist.targets[:50000]]
        mnist_val = [mnist.data[50000:], mnist.targets[50000:]]
        idx = np.random.permutation(range(len(mnist_train[1])))
        
        mnist_train[0] = mnist_train[0][idx]
        mnist_train[1] = mnist_train[1][idx]
        
        eta, beta, delta = hparams['cmnist_eta'], hparams['cmnist_beta'], hparams['cmnist_delta']
        
        self.sets = {'e1': {
                'images': mnist_train[0][::2],
                'labels': mnist_train[1][::2]
            },            
            'e2':{
                'images': mnist_train[0][1::2],
                'labels': mnist_train[1][1::2]
            },
            'val':{
                'images': mnist_val[0],
                'labels': mnist_val[1]
            }
           }
    
        self.ps = {'e1': beta + (delta / 2),
              'e2': beta - (delta / 2),
              'val': 0.9}
        
        for s in self.sets:
            imgs = self.sets[s]['images']
            labels = self.sets[s]['labels']
            # 2x subsample for computational convenience
            imgs =  imgs.reshape((-1, 28, 28))[:, ::2, ::2]
            imgs = torch.stack([imgs, imgs], dim = 1)        

            labels = torch_xor((labels < 5).float(),
                                          torch_bernoulli(eta, len(labels)))

            colors = torch_xor(labels, torch_bernoulli(self.ps[s], len(labels)))
            imgs[torch.tensor(range(len(imgs))), (1-colors).long(), :, :] *= 0

            self.sets[s]['images'] = imgs.float()/255.
            self.sets[s]['images'] = self.sets[s]['images'].reshape(self.sets[s]['images'].shape[0], -1)
            self.sets[s]['labels'] = labels.squeeze().long()    


def torch_bernoulli(p, size):
    return (torch.rand(size) < p).float()
    
def torch_xor(a, b):
    return (a-b).abs() # Assumes both inputs are either 0 or 1   