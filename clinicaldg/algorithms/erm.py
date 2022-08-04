import numpy as np

import torch
import torch.nn as nn

from clinicaldg.lib.evalution import cross_entropy

from .base import Algorithm
from .utils import cat

class ERM(Algorithm):
    """
    Empirical Risk Minimization (ERM)
    """

    def __init__(self, experiment, num_domains, hparams):
        super(ERM, self).__init__(experiment, num_domains, hparams)
        self.featurizer = experiment.get_featurizer(self.hparams)
        self.classifier = nn.Linear(self.featurizer.n_outputs, experiment.num_classes)
        self.network = nn.Sequential(self.featurizer, self.classifier)
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay']
        )       

    def update(self, minibatches, device):
        all_x = cat([x for x,y in minibatches])
        all_y = torch.cat([y for x,y in minibatches])
        loss = cross_entropy(self.predict(all_x), all_y.squeeze().long())

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {'loss': loss.item()}

    def predict(self, x):
        return self.network(x)


class ERMID(ERM):
    pass

class ERMMerged(ERM):
    pass