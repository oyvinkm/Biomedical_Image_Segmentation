from typing import Callable, Optional, Sequence, Union
from torch import nn
import torch
from torch.functional import Tensor
from torch.nn import Softmax
import numpy as np
from torch.nn.modules.loss import _Loss

class DiceLoss(_Loss):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()
    def get_fields(self):
        return {'empty' : 0}
    def forward(self, inputs, targets, smooth=1e-5):
        inputs = torch.sigmoid(inputs)
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        intersection = (inputs*targets).sum()
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)
        return 1 - dice

class TverskyLoss(nn.Module):
    """https://www.kaggle.com/bigironsphere/loss-function-library-keras-pytorch"""
    def __init__(self, weight : tuple = (0.5, 0.5)):
        super(TverskyLoss, self).__init__()
        self.alpha = weight[0]
        self.beta = weight[1]
    
    def get_name(self):
        return "TverskyLoss"

    def get_fields(self):
        return {'alpha': self.alpha,
                'beta': self.beta}

    def forward(self, inputs, targets, smooth=0):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        #inputs = torch.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        #True Positives, False Positives & False Negatives
        TP = (inputs * targets).sum()    
        FP = ((1-targets) * inputs).sum()
        FN = (targets * (1-inputs)).sum()
       
        Tversky = (TP + smooth) / (TP + self.alpha*FP + self.beta*FN + smooth)  
        
        return 1 - Tversky

class WeightedDiceLoss(_Loss):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1e-5):
        inputs = torch.sigmoid(inputs)
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        intersection = (inputs*targets).sum()
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)
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
        loss = ((input*(target)).sum())/((input*(target)).sum()+self.alpha*((input*abs(target - 1)).sum()) + self.beta*(((1-input)*(target)).sum()))

        return 1 - loss



class BinaryFocalLoss(_Loss):
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

    def __init__(self, alpha=.3, gamma=2, ignore_index=None, reduction='mean', **kwargs):
        super(BinaryFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.smooth = 1e-6  # set '1e-4' when train with FP16
        self.ignore_index = ignore_index
        self.reduction = reduction

        assert self.reduction in ['none', 'mean', 'sum']
    def get_fields(self):
        return {'alpha': self.alpha, 
                'gamma': self.gamma}
    def forward(self, output, target):
        #prob = torch.sigmoid(output)
        prob = torch.clamp(output, self.smooth, 1.0 - self.smooth)

        pos_mask = (target == 1).float()
        neg_mask = (target == 0).float()

        probs_neg = 1 - prob
        pos_weight = (pos_mask * torch.pow(1 - prob, self.gamma))
        pos_loss = -self.alpha * pos_weight * nn.LogSigmoid()(prob)  

        neg_weight = (neg_mask * torch.pow(prob, self.gamma))
        neg_loss = -(1-self.alpha )* neg_weight * nn.LogSigmoid()(probs_neg)
        loss_tmp = pos_loss + neg_loss
        if self.reduction == 'none':
            loss = loss_tmp
        elif self.reduction == 'mean':
            loss = torch.mean(loss_tmp)
        elif self.reduction == 'sum':
            loss = torch.sum(loss_tmp)
        else:
            raise NotImplementedError(f"Invalid reduction mode: {self.reduction}")
        return loss


class DiceFocalLoss(nn.Module):
    """
    Compute both Dice loss and Focal Loss, and return the weighted sum of these two losses.
    The details of Dice loss is shown in ``monai.losses.DiceLoss``.
    The details of Focal Loss is shown in ``monai.losses.FocalLoss``.

    """

    def __init__(
        self,
        reduction: str = "mean",
        lambda_dice: float = 1.0,
        lambda_focal: float = 1.0,
    ) -> None:
        """
        Args:
            ``gamma``, ``focal_weight`` and ``lambda_focal`` are only used for focal loss.
            ``include_background``, ``to_onehot_y``and ``reduction`` are used for both losses
            and other parameters are only used for dice loss.
            include_background: if False channel index 0 (background category) is excluded from the calculation.
            to_onehot_y: whether to convert `y` into the one-hot format. Defaults to False.
            sigmoid: if True, apply a sigmoid function to the prediction, only used by the `DiceLoss`,
                don't need to specify activation function for `FocalLoss`.
            softmax: if True, apply a softmax function to the prediction, only used by the `DiceLoss`,
                don't need to specify activation function for `FocalLoss`.
            other_act: if don't want to use `sigmoid` or `softmax`, use other callable function to execute
                other activation layers, Defaults to ``None``. for example: `other_act = torch.tanh`.
                only used by the `DiceLoss`, don't need to specify activation function for `FocalLoss`.
            squared_pred: use squared versions of targets and predictions in the denominator or not.
            jaccard: compute Jaccard Index (soft IoU) instead of dice or not.
            reduction: {``"none"``, ``"mean"``, ``"sum"``}
                Specifies the reduction to apply to the output. Defaults to ``"mean"``.

                - ``"none"``: no reduction will be applied.
                - ``"mean"``: the sum of the output will be divided by the number of elements in the output.
                - ``"sum"``: the output will be summed.

            smooth_nr: a small constant added to the numerator to avoid zero.
            smooth_dr: a small constant added to the denominator to avoid nan.
            batch: whether to sum the intersection and union areas over the batch dimension before the dividing.
                Defaults to False, a Dice loss value is computed independently from each item in the batch
                before any `reduction`.
            gamma: value of the exponent gamma in the definition of the Focal loss.
            focal_weight: weights to apply to the voxels of each class. If None no weights are applied.
                The input can be a single value (same weight for all classes), a sequence of values (the length
                of the sequence should be the same as the number of classes).
            lambda_dice: the trade-off weight value for dice loss. The value should be no less than 0.0.
                Defaults to 1.0.
            lambda_focal: the trade-off weight value for focal loss. The value should be no less than 0.0.
                Defaults to 1.0.

        """
        super().__init__()
        self.dice = DiceLoss()
        self.focal = BinaryFocalLoss()
        if lambda_dice < 0.0:
            raise ValueError("lambda_dice should be no less than 0.0.")
        if lambda_focal < 0.0:
            raise ValueError("lambda_focal should be no less than 0.0.")
        self.lambda_dice = lambda_dice
        self.lambda_focal = lambda_focal
    def get_fields(self):
        return {'dunno': 0}
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input: the shape should be BNH[WD]. The input should be the original logits
                due to the restriction of ``monai.losses.FocalLoss``.
            target: the shape should be BNH[WD] or B1H[WD].

        Raises:
            ValueError: When number of dimensions for input and target are different.
            ValueError: When number of channels for target is neither 1 nor the same as input.

        """
        if len(input.shape) != len(target.shape):
            raise ValueError("the number of dimensions for input and target should be the same.")

        dice_loss = self.dice(input, target)
        focal_loss = self.focal(input, target)
        total_loss: torch.Tensor = self.lambda_dice * dice_loss + self.lambda_focal * focal_loss

        return total_loss

class FocalTversky(nn.Module):
    def __init__(self, weight : tuple=(0.5, 0.5), 
                gamma : float = 3, alpha : float = 3, 
                reduction = 'mean', 
                lamda_focal : float = 1., lambda_tversky : float = 1.):
        super(FocalTversky, self).__init__()
        self.reduction = reduction
        self.lambda_focal = lamda_focal
        self.lambda_tversky = lambda_tversky
        self.focal = BinaryFocalLoss(alpha = alpha, gamma = gamma)
        self.tversky = TverskyLoss(weight = weight)
    def get_fields(self):
        return {'dunno': 0}
    def forward(self, output, target):
        focal_loss = self.focal(output, target)
        tversky_loss = self.tversky(output, target)
        total_loss = torch.Tensor([self.lambda_focal * focal_loss, self.lambda_tversky * tversky_loss])
        if self.reduction == 'mean':
            total_loss = torch.mean(total_loss)
        elif self.reduction == 'sum':
            total_loss = torch.sum(total_loss)
        total_loss.requires_grad = True
        return total_loss

class BCEWLLoss(nn.BCEWithLogitsLoss):
    def __init__(self, weight: Optional[Tensor] = None, size_average=None, reduce=None, reduction: str = 'mean', pos_weight: Optional[Tensor] = None) -> None:
        super().__init__(weight=weight, size_average=size_average, reduce=reduce, reduction=reduction, pos_weight=pos_weight)

    def get_fields(self):
        return {'empty' : 0}