from typing import Optional

from torch import Tensor
import torch.nn.functional as F
import torch.nn as nn


def masked_bce_with_logits(logits, y, mask, reduction='mean', pos_weight=None, **kwargs):
    logits = logits[..., -1]
    logits, y, mask = logits.view(-1), y.view(-1), mask.view(-1)
    
    elem_loss = F.binary_cross_entropy_with_logits(
        logits, 
        y, 
        mask.type(y.dtype),
        pos_weight=pos_weight,
        reduction='none',
        **kwargs
    )

    # Aggregate as needed
    if reduction == 'mean':
        return elem_loss.sum() / mask.sum()
    elif reduction == 'sum':
        return elem_loss.sum()
    elif reduction == 'none': 
        return elem_loss
    else:
        raise ValueError(f"reduction must be 'mean', 'sum', or 'none', got {reduction}")


class MaskedBCEWithLogitsLoss(nn.modules.loss._Loss):
    def __init__(self, size_average=None, reduce=None, reduction: str = 'mean',
                 pos_weight=None):
        super(MaskedBCEWithLogitsLoss, self).__init__(size_average, reduce, reduction)
        self.register_buffer('pos_weight', pos_weight)
        self.pos_weight: Optional[Tensor]

    def forward(self, input, target, mask):
        return masked_bce_with_logits(input, target, mask,
                               pos_weight=self.pos_weight,
                               reduction=self.reduction)