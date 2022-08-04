import torch

from clinicaldg.lib.hparams_registry import HparamSpec
from clinicaldg.lib.evalution import cross_entropy

from .erm import ERM


class GroupDRO(ERM):
    """
    Robust ERM minimizes the error at the worst minibatch
    Algorithm 1 from [https://arxiv.org/pdf/1911.08731.pdf]
    """

    HPARAM_SPEC = ERM.HPARAM_SPEC + [
        HparamSpec('groupdro_eta', 1e-2, lambda r: 10**r.uniform(-3, -1)),
    ]

    def __init__(self, experiment, num_domains, hparams):
        super(GroupDRO, self).__init__(experiment, num_domains, hparams)
        self.register_buffer("q", torch.ones(num_domains))

    def update(self, minibatches, device):
        assert len(minibatches) == len(self.q), str(len(minibatches)) + ' ' + str(len(self.q))
        if str(self.q.device) != device:
            self.q = self.q.to(device)

        losses = torch.zeros(len(minibatches)).to(device)

        for m in range(len(minibatches)):
            x, y = minibatches[m]
            losses[m] = cross_entropy(self.predict(x), y)
            self.q[m] *= (self.hparams["groupdro_eta"] * losses[m].data).exp()

        self.q /= self.q.sum()

        loss = torch.dot(losses, self.q) / len(minibatches)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {'loss': loss.item()}