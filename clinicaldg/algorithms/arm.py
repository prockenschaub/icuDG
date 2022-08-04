import torch

from .erm import ERM


class ARM(ERM):
    """ Adaptive Risk Minimization (ARM) """
    
    # PR: this algorithm isn't currently useable (and isn't reported in the 
    #     original manuscript by Zhang et al.) since networks.ContextNet isn't
    #     currently implemented.

    def __init__(self, experiment, num_domains, hparams):
        raise NotImplementedError()

        original_input_shape = experiment.input_shape
        input_shape = (1 + original_input_shape[0],) + original_input_shape[1:]
        super(ARM, self).__init__(input_shape, experiment.num_classes, num_domains,
                                  hparams)
        self.context_net = networks.ContextNet(original_input_shape) # <- this does not currently exist
        self.support_size = hparams['batch_size']

    def predict(self, x):
        batch_size, c, h, w = x.shape
        if batch_size % self.support_size == 0:
            meta_batch_size = batch_size // self.support_size
            support_size = self.support_size
        else:
            meta_batch_size, support_size = 1, batch_size
        context = self.context_net(x)
        context = context.reshape((meta_batch_size, support_size, 1, h, w))
        context = context.mean(dim=1)
        context = torch.repeat_interleave(context, repeats=support_size, dim=0)
        x = torch.cat([x, context], dim=1)
        return self.network(x)

