import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class FocalLoss(nn.Module):
    "Non weighted version of Focal Loss"
    def __init__(self, gamma=2):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        
        w = [0.21552, 0.13986, 0.15949, 0.10638, 0.18484, 
            0.27322, 0.20790, 0.07077, 0.13055, 0.08299, 
             0.09116, 0.06061, 0.09980, 0.11905, 0.25253]
        self.alpha = torch.tensor(w).cuda() 

    def forward(self, inputs, targets):
        CE_loss = F.cross_entropy(inputs, targets, reduction="none")
        targets = targets.type(torch.long)
        at = self.alpha.gather(0, targets.data.view(-1))
        pt = torch.exp(-CE_loss)
        F_loss = at*(1-pt)**self.gamma * CE_loss
        return F_loss.mean()


if __name__ == "__main__":
    pass
