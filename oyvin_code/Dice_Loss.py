from torch import nn
import torch

class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()


    def forward(self, inputs, targets, smooth=1):
        print(inputs.shape)
        inputs = torch.sigmoid(inputs)
        print(inputs.max())
        inputs = inputs.view(-1)
        
        targets = targets.view(-1)

        intersection = (inputs*targets).sum()
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)

        return 1 - dice
