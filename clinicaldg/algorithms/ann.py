import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd

from clinicaldg.networks import MLP
from clinicaldg.lib.hparams_registry import HparamSpec
from clinicaldg.lib.misc import cat

from .base import Algorithm


class AbstractDANN(Algorithm):
    """Domain-Adversarial Neural Networks (abstract class)"""

    HPARAM_SPEC = Algorithm.HPARAM_SPEC + [
        HparamSpec('ann_lambda', 1.0, lambda r: 10**r.uniform(-2, 2)),
        HparamSpec('ann_weight_decay_d', 0., lambda r: 10**r.uniform(-6, -2)),
        HparamSpec('ann_weight_decay_g', 0., lambda r: 10**r.uniform(-6, -2)),
        HparamSpec('ann_d_steps_per_g_step', 1, lambda r: int(2**r.uniform(0, 3))),
        HparamSpec('ann_grad_penalty', 0., lambda r: 10**r.uniform(-2, 1)),
        HparamSpec('ann_beta1', 0.5, lambda r: r.choice([0., 0.5])),
        HparamSpec('ann_mlp_width', 256, lambda r: int(2 ** r.uniform(6, 10))),
        HparamSpec('ann_mlp_depth', 3, lambda r: int(r.choice([3, 4, 5]))),
        HparamSpec('ann_mlp_dropout', 0., lambda r: r.choice([0., 0.1, 0.5])),
        HparamSpec('ann_lr_g', 5e-5, lambda r: 10**r.uniform(-5, -3.5)),
        HparamSpec('ann_lr_d', 5e-5, lambda r: 10**r.uniform(-5, -3.5)),
    ]

    def __init__(self, experiment, num_domains, hparams, conditional, 
                    class_balance):
        super(AbstractDANN, self).__init__(experiment, num_domains, hparams)

        self.register_buffer('update_count', torch.tensor([0]))
        self.conditional = conditional
        self.class_balance = class_balance

        self.experiment = experiment
        self.loss_fn = experiment.get_loss_fn()

        # Algorithms
        self.featurizer = experiment.get_featurizer(hparams)
        self.classifier = nn.Linear(self.featurizer.n_outputs, 
            experiment.num_classes)
        self.discriminator = MLP(
            self.featurizer.n_outputs,
            hparams['ann_mlp_width'],
            hparams['ann_mlp_depth'],
            num_domains, 
            hparams['ann_mlp_dropout']
        )
        self.class_embeddings = nn.Embedding(
            experiment.num_classes+1,  
            self.featurizer.n_outputs, 
            padding_idx=experiment.num_classes # allow embedding of masked steps
        )

        # Optimizers
        self.optimizer = {
            'disc': torch.optim.Adam(
                (list(self.discriminator.parameters()) + 
                    list(self.class_embeddings.parameters())),
                lr=self.hparams['ann_lr_d'],
                weight_decay=self.hparams['ann_weight_decay_d'],
                betas=(self.hparams['ann_beta1'], 0.9)
            ), 
            'gen': torch.optim.Adam(
                (list(self.featurizer.parameters()) + 
                    list(self.classifier.parameters())),
                lr=self.hparams['ann_lr_g'],
                weight_decay=self.hparams['ann_weight_decay_g'],
                betas=(self.hparams['ann_beta1'], 0.9)
            )
        }

    def update(self, minibatches, device):
        self.update_count += 1
        all_x = cat([x for x, y in minibatches])
        all_y = cat([y for x, y in minibatches])
        all_masks = cat([self.experiment.get_mask(batch) for batch in minibatches])
        all_z = self.featurizer(all_x)
        if self.conditional:
            disc_input = all_z + self.class_embeddings(all_y.long())
        else:
            disc_input = all_z
        disc_out = self.discriminator(disc_input)
        disc_labels = torch.cat([
            torch.full(y.shape, i, dtype=torch.int64, device=device)
            for i, (x, y) in enumerate(minibatches)
        ])

        if self.class_balance:
            y_counts = F.one_hot(all_y.long().flatten()).sum(dim=0)
            weights = 1. / (y_counts[all_y.long().flatten()] * self.experiment.num_classes).float()
        else:
            weights = torch.ones_like(all_y.flatten())
        disc_loss = F.cross_entropy(disc_out.flatten(end_dim=-2), disc_labels.flatten(), reduction='none')
        disc_loss = (all_masks.flatten() * weights.flatten() * disc_loss).sum()

        input_grad = autograd.grad(
            F.cross_entropy(disc_out.flatten(end_dim=-2), disc_labels.flatten(), reduction='sum'),
            [disc_input], create_graph=True)[0]
        grad_penalty = (input_grad**2).sum(dim=-1).mean()
        disc_loss += self.hparams['ann_grad_penalty'] * grad_penalty

        d_steps_per_g = self.hparams['ann_d_steps_per_g_step']
        if (self.update_count.item() % (1+d_steps_per_g) < d_steps_per_g):

            self.optimizer['disc'].zero_grad()
            disc_loss.backward()
            self.optimizer['disc'].step()
            return {'disc_loss': disc_loss.item()}
        else:
            all_preds = self.classifier(all_z)
            classifier_loss = self.loss_fn(all_preds, all_y, all_masks)
            gen_loss = (classifier_loss +
                        (self.hparams['ann_lambda'] * -disc_loss))
            self.optimizer['disc'].zero_grad()
            self.optimizer['gen'].zero_grad()
            gen_loss.backward()
            self.optimizer['gen'].step()
            return {'gen_loss': gen_loss.item()}

    def predict(self, x):
        return self.classifier(self.featurizer(x))

class DANN(AbstractDANN):
    """Unconditional DANN"""
    def __init__(self, experiment, num_domains, hparams):
        super(DANN, self).__init__(experiment, num_domains,
            hparams, conditional=False, class_balance=False)


class CDANN(AbstractDANN):
    """Conditional DANN"""
    def __init__(self, experiment, num_domains, hparams):
        super(CDANN, self).__init__(experiment, num_domains,
            hparams, conditional=True, class_balance=True)
