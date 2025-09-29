import torch
from utils.losses import DiceLoss, BCEWithLogits, DiceBCE

def test_losses_shapes():
    logits = torch.randn(2,3,32,32)
    target = (torch.rand(2,3,32,32)>0.5).float()
    for L in [DiceLoss(), BCEWithLogits(), DiceBCE()]:
        loss = L(logits, target)
        assert loss.shape==() or isinstance(loss.item(), float)
