from functools import partial
import torch

from clinicaldg.lib.misc import predict_on_set
from clinicaldg.lib.metrics import roc_auc_score
from clinicaldg.lib.losses import masked_bce_with_logits


class BinaryClassificationMixin:
    def get_loss_fn(self):
        pass

    def get_mask(self, batch):
        # batch: x, y, ...
        y = batch[1]
        return torch.ones_like(y, dtype=torch.bool)

class BinaryTSClassficationMixin(BinaryClassificationMixin):
    def get_loss_fn(self):
        if hasattr(self, "case_weight"):
            pos_weight = self.case_weight
        else:
            pos_weight = None
        return partial(masked_bce_with_logits, pos_weight=pos_weight)
