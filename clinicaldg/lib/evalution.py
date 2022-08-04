import numpy as np
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

def cross_entropy(logits, y):
    # multiclass
    if y.ndim == 1 or y.shape[1] == 1:
        return F.cross_entropy(logits, y)
    # multitask
    else:
        return F.binary_cross_entropy_with_logits(logits, y.float())

def binary_clf_metrics(preds, targets, grp, env_name, mask = None):
    if mask is not None:
        preds = preds[mask]
        targets = targets[mask]
        grp = grp[mask]
    preds_rounded = np.round(preds)
    opt_thres = compute_opt_thres(targets, preds)

    preds_rounded_opt = (preds >= opt_thres)
    tpr_gap_opt = recall_score(targets[grp], preds_rounded_opt[grp], zero_division = 0) - recall_score(targets[~grp], preds_rounded_opt[~grp], zero_division = 0)
    tnr_gap_opt = tnr(targets[grp], preds_rounded_opt[grp]) - tnr(targets[~grp], preds_rounded_opt[~grp])
    parity_gap_opt = (preds_rounded_opt[grp].sum() / grp.sum()) - (preds_rounded_opt[~grp].sum() / (~grp).sum())    
    phi_opt = matthews_corrcoef(preds_rounded_opt, grp)
    
    return {env_name + '_roc': roc_auc_score(targets, preds),
           env_name + '_acc': accuracy_score(targets, preds_rounded_opt),
           env_name + '_tpr_gap': tpr_gap_opt,
           env_name + '_tnr_gap': tnr_gap_opt,
           env_name + '_parity_gap': parity_gap_opt,
           env_name + '_phi': phi_opt,}
