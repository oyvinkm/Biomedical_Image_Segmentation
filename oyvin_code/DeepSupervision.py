from torch import nn


class DeepSupervisionLoss(nn.Module):
    def __init__(self, loss, weigths_factors = None):
        super(DeepSupervisionLoss, self).__init__()
        self.weigths_factors = weigths_factors
        self.loss = loss


    def forward(self, x, y):
        if self.weigths_factors is None:
            weights = [1] * len(x)
        else:
            weights = self.weigths_factors
        l = weights[0] * self.loss(x[0], y[0])
        for i in range(1, len(x)):
            if weights[i] != 0:
                l += weights[i] * self.loss(x[i], y[i])
        return l



