# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# Modifications made by Patrick Rockenschaub
import torch
import torch.nn as nn

from icudg.lib.misc import cat

from .base import Algorithm

class ERM(Algorithm):
    """
    Empirical Risk Minimization (ERM)
    """

    def __init__(self, task, num_domains, hparams):
        super(ERM, self).__init__(task, num_domains, hparams)
        self.featurizer = task.get_featurizer()
        self.classifier = nn.Linear(self.featurizer.n_outputs, task.num_classes)
        self.network = nn.Sequential(self.featurizer, self.classifier)
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay']
        )       

    def update(self, minibatches, device):
        all_x = cat([x for x,y in minibatches])
        all_y = cat([y for x,y in minibatches])
        all_m = cat([self.task.get_mask(batch) for batch in minibatches])
        loss = self.loss_fn(self.predict(all_x).flatten(end_dim=-2), all_y.flatten(), all_m.flatten())

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