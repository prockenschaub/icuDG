from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from backpack import backpack, extend
from backpack.extensions import BatchGrad

from icudg.lib.hparams_registry import HparamSpec
from icudg.lib.misc import MovingAverage, l2_between_dicts

from .base import Algorithm



class Fishr(Algorithm):
    "Invariant Gradients variances for Out-of-distribution Generalization"

    HPARAM_SPEC = Algorithm.HPARAM_SPEC + [
        HparamSpec('fishr_lambda', 1000., lambda r: 10**r.uniform(1., 4.)),
        HparamSpec('fishr_penalty_anneal_iters', 1500, lambda r: int(r.uniform(0., 5000.))),
        HparamSpec('fishr_ema', 0.95, lambda r: r.uniform(0.90, 0.99)),
    ]

    def __init__(self, task, num_domains, hparams):
        super(Fishr, self).__init__(task, num_domains, hparams)

        self.featurizer = task.get_featurizer()
        emb_dim = self.featurizer.n_outputs
        self.classifier = extend(
            torch.nn.Sequential(
                torch.nn.Linear(emb_dim, emb_dim // 2),
                torch.nn.ReLU(),
                torch.nn.Linear(emb_dim // 2, emb_dim // 4),
                torch.nn.ReLU(),
                torch.nn.Linear(emb_dim // 4, task.num_classes))
            )
        self.network = nn.Sequential(self.featurizer, self.classifier)

        self.register_buffer("update_count", torch.tensor([0]))
        self.loss_extended = task.get_extended_loss_fn(reduction='none') # 
        self.ema_per_domain = [
            MovingAverage(ema=self.hparams["fishr_ema"], oneminusema_correction=True)
            for _ in range(self.num_domains)
        ]
        self._init_optimizer()

    def _init_optimizer(self):
        self.optimizer = torch.optim.Adam(
            list(self.featurizer.parameters()) + list(self.classifier.parameters()),
            lr=self.hparams["lr"],
            weight_decay=self.hparams["weight_decay"],
        )

    @property
    def warmup(self):
        return self.update_count < self.hparams["fishr_penalty_anneal_iters"]

    def update(self, minibatches, device):
        assert len(minibatches) == self.num_domains
        all_x = torch.cat([x for x, y in minibatches])
        all_y = torch.cat([y for x, y in minibatches])
        all_m = self.task.get_mask((all_x, all_y))
        len_minibatches = [x.shape[0] for x, y in minibatches]

        all_z = self.featurizer(all_x)
        all_logits = self.classifier(all_z)

        penalty = self.compute_fishr_penalty(all_logits, all_y, all_m, len_minibatches)
        all_nll = self.loss_fn(all_logits.flatten(end_dim=-2), all_y.flatten(), all_m.flatten())

        penalty_weight = 0
        if self.update_count >= self.hparams["fishr_penalty_anneal_iters"]:
            penalty_weight = self.hparams["fishr_lambda"]
            if self.update_count == self.hparams["fishr_penalty_anneal_iters"] != 0:
                # Reset Adam as in IRM or V-REx, because it may not like the sharp jump in
                # gradient magnitudes that happens at this step.
                self._init_optimizer()
        self.update_count += 1

        objective = all_nll + penalty_weight * penalty
        self.optimizer.zero_grad()
        objective.backward()
        self.optimizer.step()

        return {'loss': objective.item(), 'nll': all_nll.item(), 'penalty': penalty.item()}

    def compute_fishr_penalty(self, all_logits, all_y, all_m, len_minibatches):
        dict_grads = self._get_grads(all_logits, all_y, all_m)
        grads_var_per_domain = self._get_grads_var_per_domain(dict_grads, len_minibatches)
        return self._compute_distance_grads_var(grads_var_per_domain)

    def _get_grads(self, logits, y, m):
        self.optimizer.zero_grad()
        loss = self.loss_extended(logits.flatten(end_dim=-2), y.flatten(), m.flatten()) 
        loss = loss.sum()
        with backpack(BatchGrad()):
            loss.backward(
                # inputs=list(self.classifier.parameters()), # NOTE: only works with backpack <= 1.3.0
                retain_graph=True, create_graph=True
            )

        # compute individual grads for all samples across all domains simultaneously
        dict_grads = OrderedDict(
            [
                (name, weights.grad_batch.clone().view(weights.grad_batch.size(0), -1))
                for name, weights in self.classifier.named_parameters()
            ]
        )
        return dict_grads

    def _get_grads_var_per_domain(self, dict_grads, len_minibatches):
        # grads var per domain
        grads_var_per_domain = [{} for _ in range(self.num_domains)]
        for name, _grads in dict_grads.items():
            all_idx = 0
            for domain_id, bsize in enumerate(len_minibatches):
                env_grads = _grads[all_idx:all_idx + bsize]
                all_idx += bsize
                env_mean = env_grads.mean(dim=0, keepdim=True)
                env_grads_centered = env_grads - env_mean
                grads_var_per_domain[domain_id][name] = (env_grads_centered).pow(2).mean(dim=0)

        # moving average
        for domain_id in range(self.num_domains):
            grads_var_per_domain[domain_id] = self.ema_per_domain[domain_id].update(
                grads_var_per_domain[domain_id]
            )

        return grads_var_per_domain

    def _compute_distance_grads_var(self, grads_var_per_domain):

        # compute gradient variances averaged across domains
        grads_var = OrderedDict(
            [
                (
                    name,
                    torch.stack(
                        [
                            grads_var_per_domain[domain_id][name]
                            for domain_id in range(self.num_domains)
                        ],
                        dim=0
                    ).mean(dim=0)
                )
                for name in grads_var_per_domain[0].keys()
            ]
        )

        penalty = 0
        for domain_id in range(self.num_domains):
            penalty += l2_between_dicts(grads_var_per_domain[domain_id], grads_var)
        return penalty / self.num_domains

    def predict(self, x):
        return self.network(x)