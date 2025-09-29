import torch
import numpy as np
from sklearn.metrics import roc_auc_score

def dice_from_logits(logits, target, eps=1e-6):
    probs = torch.sigmoid(logits)
    preds = (probs>0.5).float()
    inter = (preds*target).sum(dim=(2,3))
    den = preds.sum(dim=(2,3))+target.sum(dim=(2,3))+eps
    dice = (2*inter+eps)/(den+eps)
    return dice.mean().item()

def dice_iou_from_binary(gt: np.ndarray, pr: np.ndarray, eps=1e-6):
    inter = (gt*pr).sum()
    dice = (2*inter+eps)/(gt.sum()+pr.sum()+eps)
    union = gt.sum()+pr.sum()-inter
    iou = (inter+eps)/(union+eps)
    return dice, iou

def auc_roc(y_true: np.ndarray, y_prob: np.ndarray):
    return roc_auc_score(y_true, y_prob, average='macro')
