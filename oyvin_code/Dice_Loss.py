from torch import nn
import torch

class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()


    def forward(self, inputs, targets, smooth=1):
        print('input shape', inputs.shape)
        print('target shape', targets.shape)
        inputs = torch.sigmoid(inputs)
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        print('sum inputs ', inputs.sum())
        print('sum targets', targets.sum())
        intersection = (inputs*targets).sum()
        print('intersection' , intersection)
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)
        print(dice)
        return 1 - dice
