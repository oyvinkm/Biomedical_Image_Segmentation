import torch
from torch import nn, Tensor

class ExtraCrossEntropy(nn.CrossEntropyLoss):

    def forward(self, input:Tensor, target:Tensor) -> Tensor:
        if len(target.shape) == len(input.shape):
            assert target.shape[1] == 1
            target = target[:,0]
        return super().forward(input, target.long())