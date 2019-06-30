from torch.nn import Module
from torch.nn import BCELoss
import torch

################################################
##### https://arxiv.org/pdf/1801.05746.pdf #####
################################################


class BCELossWithJaccard(Module):
    def __init__(self):
        super(BCELossWithJaccard, self).__init__()

    def forward(self, x, y, eps=1e-12):
        bce = BCELoss()(x, y)
        # One-hot encode
        x[x >= 0.5] = 1.
        x[x < 0.5] = 0.
        y[y >= 0.5] = 1.
        y[y < 0.5] = 0.
        xy = x*y
        jaccard = torch.mean(xy / (x + y - xy))
        return bce - torch.log(jaccard + eps)
