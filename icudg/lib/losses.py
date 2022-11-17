from typing import Optional

import torch
from torch import Tensor
import torch.nn.functional as F
import torch.nn as nn

from backpack import extend



def reduce_masked_loss(loss, mask, reduction):
    if reduction == 'mean':
        return loss.sum() / mask.sum()
    elif reduction == 'sum':
        return loss.sum()
    elif reduction == 'none': 
        return loss
    else:
        raise ValueError(f"reduction must be 'mean', 'sum', or 'none', got {reduction}")

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

    return reduce_masked_loss(elem_loss, mask, reduction)
    

class MaskedBCEWithLogitsLoss(nn.modules.loss._Loss):
    def __init__(self, reduction: str = 'mean', pos_weight=None):
        super(MaskedBCEWithLogitsLoss, self).__init__(size_average=None, reduce=None, reduction=reduction)
        self.register_buffer('pos_weight', pos_weight)
        self.pos_weight: Optional[Tensor]

    def forward(self, input, target, mask):
        return masked_bce_with_logits(input, target, mask,
                               pos_weight=self.pos_weight,
                               reduction=self.reduction)


class MaskedExtendedBCEWithLogitsLoss(nn.Module):
    def __init__(self, reduction: str = 'mean', pos_weight=None):
        super(MaskedExtendedBCEWithLogitsLoss, self).__init__()
        if pos_weight is not None:
            weights = torch.ones((2,))
            weights[1] = pos_weight
        else:
            weights = pos_weight
        self.extended_ce = extend(nn.CrossEntropyLoss(weights, reduction=reduction))

    def forward(self, input, target, mask):
        masked_input = input[mask]
        masked_target = target[mask]
        return self.extended_ce(masked_input, masked_target)
