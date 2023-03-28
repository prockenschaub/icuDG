# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# Modifications made by Patrick Rockenschaub
import copy

import torch
import torch.autograd as autograd

from icudg.lib.hparams_registry import HparamSpec
from icudg.lib.misc import split_meta_train_test

from .erm import ERM

class MLDG(ERM):
    """Model-Agnostic Meta-Learning
    
    Implements algorithm 1 / equation (3) from: https://arxiv.org/pdf/1710.03463.pdf
    Related: https://arxiv.org/pdf/1703.03400.pdf
    Related: https://arxiv.org/pdf/1910.13580.pdf
    """
    
    HPARAM_SPEC = ERM.HPARAM_SPEC + [
        HparamSpec('mldg_beta', 1., lambda r: 10**r.uniform(-1, 1)),
        HparamSpec('n_meta_test', 2, lambda r:  int(r.choice([1, 2])))
    ]

    def __init__(self, task, num_domains, hparams):
        super(MLDG, self).__init__(task, num_domains, hparams)
        self.num_meta_test = hparams['n_meta_test']

    def update(self, minibatches, device):
        """
        Terms being computed:
            * Li = Loss(xi, yi, params)
            * Gi = Grad(Li, params)

            * Lj = Loss(xj, yj, Optimizer(params, grad(Li, params)))
            * Gj = Grad(Lj, params)

            * params = Optimizer(params, Grad(Li + beta * Lj, params))
            *        = Optimizer(params, Gi + beta * Gj)

        That is, when calling .step(), we want grads to be Gi + beta * Gj

        For computational efficiency, we do not compute second derivatives.
        """
        num_train = len(minibatches) - self.num_meta_test
        nll = 0
        penalty = 0

        self.optimizer.zero_grad()
        for p in self.network.parameters():
            if p.grad is None:
                p.grad = torch.zeros_like(p)

        for (xi, yi), (xj, yj) in split_meta_train_test(minibatches, self.num_meta_test):
            # fine tune clone-network on task "i"
            inner_net = copy.deepcopy(self.network)

            inner_opt = torch.optim.Adam(
                inner_net.parameters(),
                lr=self.hparams["lr"],
                weight_decay=self.hparams['weight_decay']
            )

            maski = self.task.get_mask((xi, yi))
            inner_obj = self.loss_fn(inner_net(xi).flatten(end_dim=-2), yi.flatten(), maski.flatten())

            inner_opt.zero_grad()
            inner_obj.backward()
            inner_opt.step()

            # The network has now accumulated gradients Gi
            # The clone-network has now parameters P - lr * Gi
            for p_tgt, p_src in zip(self.network.parameters(),
                                    inner_net.parameters()):
                if p_src.grad is not None:
                    p_tgt.grad.data.add_(p_src.grad.data / num_train)

            # `nll` is populated for reporting purposes
            nll += inner_obj.item() / num_train

            # this computes Gj on the clone-network
            maskj = self.task.get_mask((xj, yj))
            loss_inner_j = self.loss_fn(inner_net(xj).flatten(end_dim=-2), yj.flatten(), maskj.flatten())
            grad_inner_j = autograd.grad(loss_inner_j, inner_net.parameters(),
                allow_unused=True)

            # `penalty` is populated for reporting purposes
            penalty += loss_inner_j.item() / num_train

            for p, g_j in zip(self.network.parameters(), grad_inner_j):
                if g_j is not None:
                    p.grad.data.add_(
                        self.hparams['mldg_beta'] * g_j.data / num_train)

            # The network has now accumulated gradients Gi + beta * Gj
            # Repeat for all train-test splits, do .step()

        loss = nll + self.hparams['mldg_beta'] * penalty
        self.optimizer.step()

        return {'loss': loss, 'nll': nll, 'penalty': penalty}
