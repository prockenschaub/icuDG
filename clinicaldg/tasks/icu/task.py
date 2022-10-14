import numpy as np
import pandas as pd

from torch.utils.data import ConcatDataset


from clinicaldg.lib.hparams_registry import HparamSpec
from clinicaldg.lib.misc import predict_on_set, cat
from clinicaldg.lib.metrics import roc_auc_score
from clinicaldg.tasks import base, types

from . import data, featurizer


def _not(lst, excl):
    return [x for x in lst if x not in excl]


class MulticenterICU(base.Task):
    
    ENVIRONMENTS = ['mimic', 'eicu', 'hirid', 'aumc']
    
    num_classes = 2
    input_shape = None
    
    HPARAM_SPEC = [
        # Data
        HparamSpec('val_env', None),
        HparamSpec('test_env', 'mimic'),

        # Training
        HparamSpec('lr', 1e-3, lambda r: float(np.exp(r.uniform(low=-10, high=-3)))),
        HparamSpec('batch_size', 128, lambda r: int(r.choice(a=[128, 256, 512]))),

        # Network
        HparamSpec('architecture', 'tcn'),
        HparamSpec('hidden_dims', 64, lambda r: int(r.choice(a=[32, 64, 128]))),
        HparamSpec('num_layers', 1, lambda r: int(r.randint(low=1, high=10))),
        HparamSpec('kernel_size', 4, lambda r: int(r.randint(low=2, high=6))),
        HparamSpec('heads', 4, lambda r: int(r.randint(low=1, high=3))),
        HparamSpec('dropout', 0.5, lambda r: float(r.choice(a=[0.3, 0.4, 0.5, 0.6, 0.7])))

    ]

    def __init__(self, outcome, hparams, args):
        self.outcome = outcome
        self.args = args
        self.hparams = hparams
        self.envs = {e: data.Environment(e, outcome, self.pad_to) for e in self.ENVIRONMENTS}

        # Assign environments to train / val / test
        self.TRAIN_ENVS = _not(self.ENVIRONMENTS, [hparams['val_env']] + [hparams['test_env']])
        if hparams['val_env'] == 'train':
            self.VAL_ENVS = self.TRAIN_ENVS
        else:
            self.VAL_ENVS = [hparams['val_env']]
        self.TEST_ENVS = [hparams['test_env']]

    def add_environment(self, name):
        self.envs[name] = data.Environment(name, self.hparams['outcome'])

    def setup(self, envs=None, use_weight=True):
        """Perform actual data loading and preprocessing"""
        if envs is None:
            envs = [e for e in self.envs.keys()]

        for name, obj in self.envs.items():
            if name in envs:
                obj.prepare(
                    self.args.train_pct, 
                    self.args.val_pct, 
                    self.args.seed, 
                    self.args.debug
                )
        
        # Check that all have the same number of inputs
        input_dims = np.unique([e.num_inputs for e in self.envs.values() if e.loaded])
        if len(input_dims) > 1:
            raise ValueError(f'Different input dimensions in envs: {input_dims}')
        self.num_inputs = int(input_dims)

        # Calculate case weights based on train fold of train envs
        if use_weight:
            train_data = pd.concat([self.envs[e]['train'].data['outc'] for e in self.TRAIN_ENVS])
            prop_cases = np.mean(train_data.label)
            self.case_weight = (1 - prop_cases) / prop_cases
        else:
            self.case_weight = None

    def get_torch_dataset(self, envs, dset):
        return ConcatDataset([self.envs[e][dset] for e in envs])

    def get_featurizer(self, hparams):
        if hparams['architecture'] == "tcn":
            return featurizer.TCNet(
                self.num_inputs,
                hparams['hidden_dims'],
                hparams['num_layers'],
                hparams['kernel_size'],
                hparams['dropout']
            )
        elif hparams['architecture'] == "transformer":
            return featurizer.TransformerNet(
                self.num_inputs,
                hparams['hidden_dims'],
                hparams['num_layers'],
                hparams['heads'],
                hparams['dropout']
            )
        return NotImplementedError(
            f"Architecture {hparams['architecture']} not available ",
            f"as a featurizer for the MultiCenter task"
        )

    def eval_metrics(self, algorithm, loader, device, **kwargs):
        logits, y, mask = predict_on_set(algorithm, loader, device, self.get_mask)
        logits = logits[..., -1]
        mask = cat(mask)
        
        # Get the "normal" masked logits for each time step
        logits = logits.view(-1)[mask.view(-1)].numpy()
        y = y.view(-1)[mask.view(-1)].long().numpy()

        return {'roc': roc_auc_score(y, logits)}


class Mortality24(MulticenterICU, types.BinaryTSClassficationMixin):
    def __init__(self, hparams, args):
        self.pad_to = None
        super().__init__('mortality24', hparams, args)

    def get_featurizer(self, hparams):
        return featurizer.LastStep(super(Mortality24, self).get_featurizer(hparams))


class Sepsis(MulticenterICU, types.BinarySeq2SeqClassificationMixin):
    def __init__(self, hparams, args):
        self.pad_to = 193
        super().__init__("sepsis", hparams, args)

    def get_mask(self, batch):
        # batch: x, y, ...
        y = batch[1]
        return y != data.PAD_VALUE