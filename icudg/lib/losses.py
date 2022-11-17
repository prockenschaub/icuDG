import torch.nn as nn

import backpack

    

class MaskedBCEWithLogitsLoss(nn.Module):
    def __init__(self, weights=None, reduction: str = 'mean',extend=False):
        super(MaskedBCEWithLogitsLoss, self).__init__()
        self.ce_loss = nn.CrossEntropyLoss(weights, reduction=reduction)
        if extend:
            self.ce_loss = backpack.extend(self.ce_loss)

    def forward(self, input, target, mask):
        if len(input.shape) != 2:
            raise ValueError(f"expected input to have 2 dimensions, got {len(input.shape)} instead.")
        elif len(target.shape) != 1:
            raise ValueError(f"expected target to have 1 dimension, got {len(target.shape)} instead.")
        elif len(mask.shape) != 1:
            raise ValueError(f"expected mask to have 1 dimension, got {len(mask.shape)} instead.")
        if input.shape[-1] != 2:
            raise ValueError(f"the final dimension of `input` must be 2, got {input.shape[-1]} instead.")
        masked_input = input[mask]
        masked_target = target[mask]
        return self.ce_loss(masked_input, masked_target)

