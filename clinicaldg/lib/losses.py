import torch.nn.functional as F


def ts_bce_loss(logits, y, mask, reduction='mean', pos_weight=None, **kwargs):
    logits = logits[..., -1]
    y = y.squeeze(-1)
    
    if pos_weight is not None:
        pos_weight = y.new_tensor(pos_weight)

    ce = F.binary_cross_entropy_with_logits(
        logits, 
        y, 
        reduction=reduction, 
        pos_weight=pos_weight,
        **kwargs
    )

    return ce    

def seq2seq_bce_loss(logits, y, mask, reduction='mean', pos_weight=None, **kwargs):
    logits = logits[..., -1]
    if pos_weight is not None:
        pos_weight = y.new_tensor(pos_weight)

    ce = F.binary_cross_entropy_with_logits(
        logits, 
        y, 
        reduction='none', 
        pos_weight=pos_weight,
        **kwargs
    )
    
    # Mask padded values when calculating the loss
    masked_ce = ce * mask

    # Aggregate as needed
    if reduction == 'mean':
        return masked_ce.sum() / mask.sum()
    elif reduction == 'sum':
        return masked_ce.sum()
    return masked_ce