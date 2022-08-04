import torch
from torch.utils.data import TensorDataset, ConcatDataset

from clinicaldg.lib import misc
from clinicaldg.lib.hparams_registry import HparamSpec
from clinicaldg.experiments import ExperimentBase
from clinicaldg.networks import MLP

from .data import ColoredMNISTDataset


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
    
    # Define hyperparameters
    HPARAM_SPEC = [
        # Data 
        HparamSpec('cmnist_eta', 0.25),
        HparamSpec('cmnist_beta', 0.15),
        HparamSpec('cmnist_delta', 0.1),
        
        # Training
        HparamSpec('lr', 1e-3, lambda r: 10**r.uniform(-4.5, -2.5)),
        HparamSpec('batch_size', 128, lambda r: int(2**r.uniform(6, 9))),

        # Featurzier
        HparamSpec('mlp_width', 256, lambda r: int(2 ** r.uniform(6, 10))),
        HparamSpec('mlp_depth', 2, lambda r: int(r.choice([2, 3]))),
        HparamSpec('mlp_dropout', 0., lambda r: r.choice([0., 0.1])),

        # Algorithm-specific
        HparamSpec('ann_lr_g', 1e-3, lambda r: 10**r.uniform(-4.5, -2.5)),
        HparamSpec('ann_lr_d', 1e-3, lambda r: 10**r.uniform(-4.5, -2.5)),
        HparamSpec('ann_weight_decay_g', 0., lambda r: 0.),
    ]

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
        return MLP(self.input_shape[0], hparams['mlp_width'], hparams['mlp_depth'], 128, hparams['mlp_dropout'])

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
    