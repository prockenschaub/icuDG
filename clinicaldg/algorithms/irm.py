import torch
import torch.autograd as autograd

from clinicaldg.lib.hparams_registry import HparamSpec
from clinicaldg.lib.evalution import cross_entropy

from .erm import ERM
from .utils import cat


class IRM(ERM):
    """Invariant Risk Minimization"""

    HPARAM_SPEC = [
        HparamSpec('irm_lambda', 1e2, lambda r: 10**r.uniform(-1, 5)),
        HparamSpec('irm_penalty_anneal_iters', 500, lambda r: int(10**r.uniform(0, 4)))
    ]

    def __init__(self, experiment, num_domains, hparams):
        super(IRM, self).__init__(experiment, num_domains, hparams)
        self.register_buffer('update_count', torch.tensor([0]))

    @staticmethod
    def _irm_penalty(logits, y):
        device = logits.device
        scale = torch.tensor(1.).to(device).requires_grad_()
        loss_1 = cross_entropy(logits[::2] * scale, y[::2])
        loss_2 = cross_entropy(logits[1::2] * scale, y[1::2])
        grad_1 = autograd.grad(loss_1, [scale], create_graph=True)[0]
        grad_2 = autograd.grad(loss_2, [scale], create_graph=True)[0]
        result = torch.sum(grad_1 * grad_2)
        return result

    def update(self, minibatches, device):
        penalty_weight = (self.hparams['irm_lambda'] if self.update_count
                          >= self.hparams['irm_penalty_anneal_iters'] else
                          1.0)
        nll = 0.
        penalty = 0.

        all_x = cat([x for x,y in minibatches])
        all_logits = self.network(all_x)
        all_logits_idx = 0
        for i, (x, y) in enumerate(minibatches):
            logits = all_logits[all_logits_idx:all_logits_idx + y.shape[0]]
            all_logits_idx += y.shape[0]
            nll += cross_entropy(logits, y)
            penalty += self._irm_penalty(logits, y)
        nll /= len(minibatches)
        penalty /= len(minibatches)
        loss = nll + (penalty_weight * penalty)

        if self.update_count == self.hparams['irm_penalty_anneal_iters']:
            # Reset Adam, because it doesn't like the sharp jump in gradient
            # magnitudes that happens at this step.
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
