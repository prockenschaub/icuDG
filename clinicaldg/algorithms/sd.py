import torch

from clinicaldg.lib.hparams_registry import HparamSpec
from clinicaldg.lib.evalution import cross_entropy

from .erm import ERM
from .utils import cat


class SD(ERM):
    """
    Gradient Starvation: A Learning Proclivity in Neural Networks
    Equation 25 from [https://arxiv.org/pdf/2011.09468.pdf]
    """
    HPARAM_SPEC = ERM.HPARAM_SPEC + [
        HparamSpec('sd_reg', 0.1, lambda r: 10**r.uniform(-5, -1)),
    ]
    
    def __init__(self, experiment, num_domains, hparams):
        super(SD, self).__init__(experiment, num_domains, hparams)
        self.sd_reg = hparams["sd_reg"] 

    def update(self, minibatches, device):
        all_x = cat([x for x,y in minibatches])
        all_y = torch.cat([y for x,y in minibatches])
        all_p = self.predict(all_x)

        loss = cross_entropy(all_p, all_y)
        penalty = (all_p ** 2).mean()
        objective = loss + self.sd_reg * penalty

        self.optimizer.zero_grad()
        objective.backward()
        self.optimizer.step()

        return {'loss': loss.item(), 'penalty': penalty.item()}