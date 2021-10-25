from torch import nn
import torch
from torch.nn import Softmax
from torch.nn.functional import logsigmoid
import math

class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        inputs = torch.sigmoid(inputs)
        inputs = inputs.view(-1)
        
        targets = targets.view(-1)

        intersection = (inputs*targets).sum()
        dice = (2.*intersection + smooth)/(inputs.sum()**2 + targets.sum()**2 + smooth)

        return 1 - dice

class WeightedTverskyLoss(nn.Module):
    """Tversky loss function from arXiv:1803.11078v1"""
    def __init__(self, weight : tuple=(0.5, 0.5)):
        super(WeightedTverskyLoss, self).__init__()
        self.alpha = weight[0]
        self.beta = weight[1]

    def forward(self, input, target):
        input = input.view(-1)
        target = target.view(-1)
        m = Softmax(dim=0)
        input = m(input)
        
        p0 = input #probability that voxel is a lacune
        p1 = 1 - input #probability that voxel is a non-lacune
        g0 = target #1 if voxel is a lacune, 0 if voxel is a non-lacune
        g1 = abs(target - 1) #0 if voxel is a lacune, 1 if voxel is a non-lacune
        loss = ((p0*g0).sum())/((p0*g0).sum()+self.alpha*((p0*g1).sum()) + self.beta*((p1*g0).sum()))

        return 1 - loss

class _BCEWithLogitsLoss(nn.Module):
    def __init__(self):
        super(_BCEWithLogitsLoss, self).__init__()
    
    def forward(self, input, target):
        input = input.view(-1)
        target = target.view(-1)
        p1 = target*(logsigmoid(input))
        p0 = (1-target)*math.log(1-torch.sigmoid(input))
        return -1 * (p1 + p0)