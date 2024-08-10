"""Dice loss implementation"""
import torch
from torch import nn


class BinaryDiceLoss(nn.Module):
    """Dice loss of binary class

    :param smooth: A float number to smooth loss, and avoid NaN error
    :return: Loss tensor
    """
    def __init__(self,
                 smooth: int = 0, ):
        super().__init__()
        self.smooth = smooth

    def forward(self, predict, target):
        intersection = torch.sum(predict * target)
        den = torch.sum(predict*predict) + torch.sum(target*target)

        return 1 - ((2. * intersection + self.smooth) / (den + self.smooth))


class DiceLoss(nn.Module):
    """Multiclass Dice loss using one hot encode input

    :param smooth: A float number to smooth loss, and avoid NaN error
    :param weights: Classes weights
    :param ignore_index: Index of a class to ignore
    :return: Loss tensor
    """
    def __init__(self,
                 smooth: int = 0,
                 weights: torch.Tensor = None,
                 ignore_index: int = None):
        super().__init__()
        self.smooth = smooth
        self.weights = weights
        self.ignore_index = ignore_index

    def forward(self, predict, target):
        assert predict.shape == target.shape, \
            'predict & target shape do not match'
        dice = BinaryDiceLoss(self.smooth)
        total_loss = 0
        predict = torch.sigmoid(predict)

        for i in range(target.shape[1]):
            if i != self.ignore_index:
                dice_loss = dice(predict[:, i], target[:, i])
                if self.weights is not None:
                    assert self.weights.shape[0] == target.shape[1], \
                        'Expect weights shape [{}], get[{}]'.format(
                            target.shape[1], self.weights.shape[0])
                    dice_loss *= self.weights[i]
                total_loss += dice_loss

        return total_loss/target.shape[1]


export = [BinaryDiceLoss, DiceLoss]
