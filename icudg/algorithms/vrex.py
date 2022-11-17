import torch

from icudg.lib.hparams_registry import HparamSpec
from icudg.lib.misc import cat

from .erm import ERM

class VREx(ERM):
    """V-REx algorithm from http://arxiv.org/abs/2003.00688"""
    
    HPARAM_SPEC = ERM.HPARAM_SPEC + [
        HparamSpec('vrex_lambda', 1e1, lambda r: 10**r.uniform(-1, 5)),
        HparamSpec('vrex_penalty_anneal_iters', 500, lambda r: int(10**r.uniform(0, 4))),
    ]
    
    def __init__(self, task, num_domains, hparams):
        super(VREx, self).__init__(task, num_domains, hparams)
        self.register_buffer('update_count', torch.tensor([0]))

    def update(self, minibatches, device):
        if self.update_count >= self.hparams["vrex_penalty_anneal_iters"]:
            penalty_weight = self.hparams["vrex_lambda"]
        else:
            penalty_weight = 1.0
        
        nll = 0.

        all_x = cat([x for x, y in minibatches])
        all_logits = self.network(all_x)
        all_logits_idx = 0
        losses = torch.zeros(len(minibatches))
        for i, (x, y) in enumerate(minibatches):
            mask = self.task.get_mask((x, y))
            logits = all_logits[all_logits_idx:all_logits_idx + y.shape[0]]
            all_logits_idx += y.shape[0]
            nll = self.loss_fn(logits, y, mask)
            losses[i] = nll

        mean = losses.mean()
        penalty = ((losses - mean) ** 2).mean()
        loss = mean + penalty_weight * penalty

        if self.update_count == self.hparams['vrex_penalty_anneal_iters']:
            # Reset Adam (like IRM), because it doesn't like the sharp jump in
            # gradient magnitudes that happens at this step.
            self.optimizer = torch.optim.Adam(
                self.network.parameters(),
                lr=self.hparams["lr"],
                weight_decay=self.hparams['weight_decay'])

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.update_count += 1
        return {'loss': loss.item(), 'nll': nll.item(),
                'penalty': penalty.item()}

    @property
    def warmup(self):
        return self.update_count < self.hparams["vrex_penalty_anneal_iters"]