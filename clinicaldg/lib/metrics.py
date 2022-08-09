import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    roc_auc_score, 
    accuracy_score, 
    recall_score, 
    f1_score, 
    confusion_matrix, 
    matthews_corrcoef
)

def compute_opt_thres(target, pred):
    opt_thres = 0
    opt_f1 = 0
    for i in np.arange(0.05, 0.9, 0.01):
        f1 = f1_score(target, pred >= i)
        if f1 >= opt_f1:
            opt_thres = i
            opt_f1 = f1
    return opt_thres

def tnr(target, pred):
    CM = confusion_matrix(target, pred, labels=[0, 1])
    
    TN = CM[0][0]
    FN = CM[1][0]
    TP = CM[1][1]
    FP = CM[0][1]
    
    return TN/(TN + FP) if (TN + FP) > 0 else 0

def cross_entropy(logits, y, **kwargs):
    # multiclass
    if y.ndim == 1 or y.shape[1] == 1:
        return F.cross_entropy(logits, y, **kwargs)
    # multitask
    else:
        return F.binary_cross_entropy_with_logits(logits, y.float(), **kwargs)


def accuracy(network, loader, weights, device):
    correct = 0
    total = 0
    weights_offset = 0

    network.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device).squeeze().long()
            p = network.predict(x)
            if weights is None:
                batch_weights = torch.ones(len(x))
            else:
                batch_weights = weights[weights_offset : weights_offset + len(x)]
                weights_offset += len(x)
            batch_weights = batch_weights.to(device)
            if p.size(1) == 1:
                correct += (p.gt(0).eq(y).float() * batch_weights.view(-1, 1)).sum().item()
            else:
                correct += (p.argmax(1).eq(y).float() * batch_weights).sum().item()
            total += batch_weights.sum().item()
    network.train()
    
    return correct / total