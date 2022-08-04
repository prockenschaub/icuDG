import torch
import torch.nn as nn

from clinicaldg.lib.hparams_registry import HparamSpec
from clinicaldg.lib.evalution import cross_entropy

from .base import Algorithm


class MTL(Algorithm):
    """
    A neural network version of
    Domain Generalization by Marginal Transfer Learning
    (https://arxiv.org/abs/1711.07910)
    """

    HPARAM_SPEC = Algorithm.HPARAM_SPEC + [
        HparamSpec('mtl_ema', .99, lambda r: r.choice([0.5, 0.9, 0.99, 1.])),
    ]

    def __init__(self, experiment, num_domains, hparams):
        super(MTL, self).__init__(experiment, num_domains, hparams)
        self.featurizer = experiment.get_featurizer(hparams)
        self.classifier = nn.Linear(self.featurizer.n_outputs * 2, 
            experiment.num_classes)
        self.optimizer = torch.optim.Adam(
            list(self.featurizer.parameters()) +\
            list(self.classifier.parameters()),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay']
        )

        self.register_buffer('embeddings',
                             torch.zeros(num_domains,
                                         self.featurizer.n_outputs))

        self.ema = self.hparams['mtl_ema']

    def update(self, minibatches, device):
        loss = 0
        for env, (x, y) in enumerate(minibatches):
            loss += cross_entropy(self.predict(x, env), y)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {'loss': loss.item()}

    def update_embeddings_(self, features, env=None):
        return_embedding = features.mean(0)

        if env is not None:
            return_embedding = self.ema * return_embedding +\
                               (1 - self.ema) * self.embeddings[env]

            self.embeddings[env] = return_embedding.clone().detach()

        return return_embedding.view(1, -1).repeat(len(features), 1)

    def predict(self, x, env=None):
        features = self.featurizer(x)
        embedding = self.update_embeddings_(features, env).normal_()
        return self.classifier(torch.cat((features, embedding), 1))