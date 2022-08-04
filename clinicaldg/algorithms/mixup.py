import numpy as np

from clinicaldg.lib.hparams_registry import HparamSpec
from clinicaldg.lib.evalution import cross_entropy
from clinicaldg.lib.misc import random_pairs_of_minibatches

from .erm import ERM


class Mixup(ERM):
    """
    Mixup of minibatches from different domains
    https://arxiv.org/pdf/2001.00677.pdf
    https://arxiv.org/pdf/1912.01805.pdf
    """
    HPARAM_SPEC = ERM.HPARAM_SPEC + [
        HparamSpec('mixup_alpha', 0.2, lambda r: 10**r.uniform(-1, -1)),
    ]

    def __init__(self, experiment, num_domains, hparams):
        super(Mixup, self).__init__(experiment, num_domains, hparams)

    def update(self, minibatches, device):
        objective = 0

        for (xi, yi), (xj, yj) in random_pairs_of_minibatches(minibatches):
            lam = np.random.beta(self.hparams["mixup_alpha"],
                                 self.hparams["mixup_alpha"])

            x = lam * xi + (1 - lam) * xj
            predictions = self.predict(x)

            objective += lam * cross_entropy(predictions, yi)
            objective += (1 - lam) * cross_entropy(predictions, yj)

        objective /= len(minibatches)

        self.optimizer.zero_grad()
        objective.backward()
        self.optimizer.step()

        return {'loss': objective.item()}