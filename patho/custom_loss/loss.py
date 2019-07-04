from torch.nn import Module
from torch.nn import BCELoss
import torch

################################################
##### https://arxiv.org/pdf/1801.05746.pdf #####
################################################


class BCELossWithJaccard(Module):
    def __init__(self):
        super(BCELossWithJaccard, self).__init__()
        self.bce = BCELoss()

    def forward(self, x, y):
        bce = self.bce(x, y)
        xy = x*y
        jaccard = torch.mean(xy / (x + y - xy))
        return bce - torch.log(jaccard + 1e-12)
