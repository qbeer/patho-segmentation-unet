from torch.nn import Module
from torch.nn import BCELoss
import torch

################################################
##### https://arxiv.org/pdf/1801.05746.pdf #####
################################################


class BCELossWithJaccard(Module):
    def __init__(self):
        super(BCELossWithJaccard, self).__init__()

    def forward(self, x, y):
        bce = BCELoss()(x, y)
        # One-hot encode
        x_ = 1. if x[x >= 0.5] else 0.
        y_ = 1. if y[y >= 0.5] else 0.
        xy_ = x_*y_
        jaccard = torch.mean(xy_ / (x_ + y_ - xy_))
        return bce - torch.log(jaccard + 1e-12)
