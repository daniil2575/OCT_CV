import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__(); self.eps=eps
    def forward(self, logits, target):
        probs = torch.sigmoid(logits)
        num = 2*(probs*target).sum(dim=(2,3))
        den = probs.sum(dim=(2,3))+target.sum(dim=(2,3))+self.eps
        return (1 - (num+self.eps)/(den+self.eps)).mean()

class BCEWithLogits(nn.Module):
    def __init__(self, pos_weight=1.0):
        super().__init__(); self.w = torch.tensor([pos_weight])
    def forward(self, logits, target):
        w = self.w.to(logits.device)
        return F.binary_cross_entropy_with_logits(logits, target, pos_weight=w)

class DiceBCE(nn.Module):
    def __init__(self, pos_weight=1.0):
        super().__init__(); self.d=DiceLoss(); self.b=BCEWithLogits(pos_weight)
    def forward(self, logits, target):
        return self.d(logits, target)+self.b(logits, target)

def build_loss(cfg):
    t = cfg['type']
    if t=='dice': return DiceLoss()
    if t=='bce': return BCEWithLogits(cfg.get('pos_weight',1.0))
    if t=='dice_bce': return DiceBCE(cfg.get('pos_weight',1.0))
    raise ValueError(t)
