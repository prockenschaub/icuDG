import torch
from torch.utils.data import TensorDataset, ConcatDataset

from clinicaldg.experiments import ExperimentBase
from clinicaldg.lib import misc
from clinicaldg.mnist.data import ColoredMNISTDataset
from clinicaldg.models import MLP


class ColoredMNIST(ExperimentBase):
    '''
    Hyperparameters:
    cmnist_eta
    cmnist_beta
    cmnist_delta   
    
    '''
    ENVIRONMENTS = ['e1', 'e2', 'val']
    TRAIN_PCT = 0.7
    VAL_PCT = 0.1
    MAX_STEPS = 1500
    N_WORKERS = 1
    CHECKPOINT_FREQ = 500 # large value to avoid test env overfitting
    ES_METRIC = 'acc'
    ES_PATIENCE = 10 # no early stopping for CMNIST to avoid test env overfitting
    TRAIN_ENVS = ['e1', 'e2']
    VAL_ENV = 'val'
    TEST_ENV = 'val'
    input_shape = (14*14*2, )
    num_classes = 2
    
    def __init__(self, hparams, args):
        super().__init__()
        self.d = ColoredMNISTDataset(hparams, args)   
    
    def get_torch_dataset(self, envs, dset):
            '''
            envs: a list of region names
            dset: split within envs, one of ['train', 'val', 'test']
            '''
            
            datasets = []
            
            for e in envs:
                xall = self.d.sets[e]['images']
                if e in self.TRAIN_ENVS:
                    if dset == 'train':
                        idx_start, idx_end = 0, int(len(xall) * self.TRAIN_PCT)
                    elif dset == 'val':
                        idx_start, idx_end = int(len(xall) * self.TRAIN_PCT), int(len(xall) * (self.TRAIN_PCT + self.VAL_PCT))
                    elif dset == 'test':
                        idx_start, idx_end = int(len(xall) * (self.TRAIN_PCT + self.VAL_PCT)), len(xall)
                    else:
                        raise NotImplementedError
                        
                elif e == self.VAL_ENV: # on validation environment, use 50% for validation and 50% for test
                    if dset == 'val':
                        idx_start, idx_end = 0, int(len(xall) * (0.5))
                    elif dset == 'test':
                        idx_start, idx_end = int(len(xall) * (0.5)), len(xall)
                    else:
                        raise NotImplementedError
                        
                datasets.append(TensorDataset(xall[idx_start:idx_end], self.sets[e]['labels'][idx_start:idx_end])) 
                
            return ConcatDataset(datasets) 

    def get_featurizer(self, hparams):
        return MLP(self.input_shape[0], 128, hparams)

    def predict_on_set(self, algorithm, loader, device):
        correct = 0
        total = 0

        algorithm.eval()
        with torch.no_grad():
            for x, y in loader:
                x = x.to(device)
                y = y.to(device).squeeze().long()
                p = algorithm.predict(x)
            
                batch_weights = torch.ones(len(x))
                batch_weights = batch_weights.to(device)

                if p.size(1) == 1:
                    correct += (p.gt(0).eq(y).float() * batch_weights.view(-1, 1)).sum().item()
                else:
                    correct += (p.argmax(1).eq(y).float() * batch_weights).sum().item()
                total += batch_weights.sum().item()
        algorithm.train()
    
        return correct / total

    def eval_metrics(self, algorithm, loader, env_name, weights, device):
        return {env_name + '_acc' : misc.accuracy(algorithm, loader, weights = None, device = device)}
    