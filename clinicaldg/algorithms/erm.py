import torch
import torch.nn as nn

from clinicaldg.lib.misc import cat

from .base import Algorithm

class ERM(Algorithm):
    """
    Empirical Risk Minimization (ERM)
    """

    def __init__(self, experiment, num_domains, hparams):
        super(ERM, self).__init__(experiment, num_domains, hparams)
        self.experiment = experiment
        self.featurizer = experiment.get_featurizer(self.hparams)
        self.classifier = nn.Linear(self.featurizer.n_outputs, experiment.num_classes)
        self.network = nn.Sequential(self.featurizer, self.classifier)
        self.loss_fn = experiment.get_loss_fn()
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay']
        )       

    def update(self, minibatches, device):
        all_x = cat([x for x,y in minibatches])
        all_y = cat([y for x,y in minibatches])
        all_m = cat([self.experiment.get_mask(batch) for batch in minibatches])
        loss = self.loss_fn(self.predict(all_x), all_y, all_m)

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