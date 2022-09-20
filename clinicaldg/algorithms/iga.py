import torch

from clinicaldg.lib.hparams_registry import HparamSpec
from clinicaldg.lib.misc import cat

from .erm import ERM


class IGA(ERM):
    """
    Inter-environmental Gradient Alignment from https://arxiv.org/pdf/2008.01883.pdf
    """
    HPARAM_SPEC = ERM.HPARAM_SPEC + [
        HparamSpec('iga_lambda', 1e3, lambda r: 10**r.uniform(-1, 5)),
    ]

    def __init__(self, experiment, num_domains, hparams):
        torch.backends.cudnn.enabled = False # GRU second order derivatives
        super(IGA, self).__init__(experiment, num_domains, hparams)
        self.register_buffer('update_count', torch.tensor([0]))

    def grad_variance_penalty(self, losses, model):
        env_num = len(losses)
        train_penalty = 0.0
        grad_avg = 0.0
        gradients = []
        for loss in losses:
            gradient = torch.autograd.grad([loss], model.parameters(), create_graph=True)
            grad_flatten = torch.cat([x.flatten() for x in gradient], dim=0)
            grad_avg += grad_flatten / env_num
            gradients.append(grad_flatten)
    
        for grad_flatten in gradients:
            train_penalty += torch.sum((grad_flatten - grad_avg)**2.0)
        return train_penalty
    
    def update(self, minibatches, device):
        penalty_weight = self.hparams["iga_lambda"]
        
        nll = 0.

        all_x = cat([x for x, y in minibatches])
        all_logits = self.network(all_x)
        all_logits_idx = 0
        losses = torch.zeros(len(minibatches))
        for i, (x, y) in enumerate(minibatches):
            mask = self.experiment.get_mask((x, y))
            logits = all_logits[all_logits_idx:all_logits_idx + y.shape[0]]
            all_logits_idx += y.shape[0]
            nll = self.loss_fn(logits, y, mask)
            losses[i] = nll

        mean = losses.mean()        
        penalty = self.grad_variance_penalty(losses, self.network)
        loss = mean + penalty_weight * penalty

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.update_count += 1
        return {'loss': loss.item(), 'nll': nll.item(),
                'penalty': penalty.item()}