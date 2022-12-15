# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# Modifications made by Patrick Rockenschaub

import torch

from icudg.lib.hparams_registry import HparamSpec

from .erm import ERM


class GroupDRO(ERM):
    """Robust ERM minimizes the error at the worst minibatch
    
    Implements algorithm 1 from https://arxiv.org/pdf/1911.08731.pdf
    """

    HPARAM_SPEC = ERM.HPARAM_SPEC + [
        HparamSpec('groupdro_eta', 1e-2, lambda r: 10**r.uniform(-3, -1)),
    ]

    def __init__(self, task, num_domains, hparams):
        super(GroupDRO, self).__init__(task, num_domains, hparams)
        self.register_buffer("q", torch.ones(num_domains))

    def update(self, minibatches, device):
        assert len(minibatches) == len(self.q), str(len(minibatches)) + ' ' + str(len(self.q))
        if str(self.q.device) != device:
            self.q = self.q.to(device)

        losses = torch.zeros(len(minibatches)).to(device)

        for m in range(len(minibatches)):
            x, y = minibatches[m]
            mask = self.task.get_mask((x, y))
            losses[m] = self.loss_fn(self.predict(x).flatten(end_dim=-2), y.flatten(), mask.flatten())
            self.q[m] *= (self.hparams["groupdro_eta"] * losses[m].data).exp()

        self.q /= self.q.sum()

        loss = torch.dot(losses, self.q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {'loss': loss.item(), 'nll': losses.mean().item()}