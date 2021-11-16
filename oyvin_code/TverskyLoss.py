import torch
from torch import nn
from torch.nn import Softmax

class WeightedTverskyLoss(nn.Module):
    """Tversky loss function from arXiv:1803.11078v1"""
    def __init__(self, weight : tuple=(0.5, 0.5)):
        super(WeightedTverskyLoss, self).__init__()
        self.alpha = weight[0]
        self.beta = weight[1]

    def forward(self, input, target):
        m = Softmax(dim=2)
        input = m(input)
        input = input.view(-1)
        target = target.view(-1)

        p0 = input #probability that voxel is a lacune
        p1 = 1 - input #probability that voxel is a non-lacune
        g0 = target #1 if voxel is a lacune, 0 if voxel is a non-lacune
        g1 = abs(target - 1) #0 if voxel is a lacune, 1 if voxel is a non-lacune
        
        loss = ((p0*g0).sum())/((p0*g0).sum()+self.alpha*((p0*g1).sum()) + self.beta*((p1*g0).sum()))

        return 1 - loss