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
        bce = BCELoss()
        loss1 = bce(x, y)
        xy_ = x_*y_
        loss2 = torch.mean(xy_ / (x_ + y_ - xy_))
        return loss1 - torch.log(loss2 + 1e-12)
