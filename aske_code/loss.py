from torch import nn
import torch
from torch.nn import Softmax
from torch.nn.functional import logsigmoid
import torch.nn.functional as F
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

class TverskyLoss(nn.Module):
    """https://www.kaggle.com/bigironsphere/loss-function-library-keras-pytorch"""
    def __init__(self, weight : tuple = (0.5, 0.5)):
        super(TverskyLoss, self).__init__()
        self.alpha = weight[0]
        self.beta = weight[1]
    
    def get_name(self):
        return "TverskyLoss"

    def forward(self, inputs, targets, smooth=0):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = torch.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        #True Positives, False Positives & False Negatives
        TP = (inputs * targets).sum()    
        FP = ((1-targets) * inputs).sum()
        FN = (targets * (1-inputs)).sum()
       
        Tversky = (TP + smooth) / (TP + self.alpha*FP + self.beta*FN + smooth)  
        
        return 1 - Tversky

class WeightedTverskyLoss(nn.Module):
    """Tversky loss function from arXiv:1803.11078v1"""
    def __init__(self, weight : tuple=(0.5, 0.5)):
        super(WeightedTverskyLoss, self).__init__()
        self.alpha = weight[0]
        self.beta = weight[1]

    def get_name(self):
        return "WeightedTversky"

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

class BinaryFocalLoss(nn.Module):
    """
    This is a implementation of Focal Loss with smooth label cross entropy supported which is proposed in
    'Focal Loss for Dense Object Detection. (https://arxiv.org/abs/1708.02002)'
        Focal_Loss= -1*alpha*(1-pt)*log(pt)
    :param alpha: (tensor) 3D or 4D the scalar factor for this criterion
    :param gamma: (float,double) gamma > 0 reduces the relative loss for well-classified examples (p>0.5) putting more
                    focus on hard misclassified example
    :param reduction: `none`|`mean`|`sum`
    :param **kwargs
        balance_index: (int) balance class index, should be specific when alpha is float
    """

    def __init__(self, alpha=3, gamma=2, ignore_index=None, reduction='mean', **kwargs):
        super(BinaryFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.smooth = 1e-6  # set '1e-4' when train with FP16
        self.ignore_index = ignore_index
        self.reduction = reduction

        assert self.reduction in ['none', 'mean', 'sum']

    def get_name(self):
        return "BinaryFocalLoss"

    def forward(self, output, target):
        #prob = torch.sigmoid(output)
        prob = torch.clamp(output, self.smooth, 1.0 - self.smooth)

        valid_mask = None
        if self.ignore_index is not None:
            valid_mask = (target != self.ignore_index).float()

        pos_mask = (target == 1)#.float()
        neg_mask = (target == 0)#.float()
        if valid_mask is not None:
            pos_mask = pos_mask * valid_mask
            neg_mask = neg_mask * valid_mask

        pos_weight = (pos_mask * torch.pow(1 - prob, self.gamma)).detach()
        pos_loss = -pos_weight * torch.log(prob)  # / (torch.sum(pos_weight) + 1e-4)

        neg_weight = (neg_mask * torch.pow(prob, self.gamma)).detach()
        neg_loss = -self.alpha * neg_weight * nn.LogSigmoid()(-output)  # / (torch.sum(neg_weight) + 1e-4)
        loss = pos_loss + neg_loss
        loss = loss.mean()
        return loss

class _BCEWithLogitsLoss(nn.Module):
    def __init__(self):
        super(_BCEWithLogitsLoss, self).__init__()
    
    def forward(self, input, target):
        input = abs(input.view(-1))
        target = abs(target.view(-1))
        m = Softmax(dim=0)
        p = m(input)
        y = m(target)
        print("math.log(p)", math.log(p))
        print("math.log(y)", math.log(y))
     
        val = y*math.log(p) + (1-y)*math.log(1-p)
        print(val)

        return val
