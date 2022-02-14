"""Define deep learning losses based on error maps

Classes
-------
SAContrarioMSELoss

"""

import math
import torch
from torch import nn

from .utils import disk_patch, ring_patch


class SAContrarioMSELoss(nn.Module):
    """Define a MSE loss weighted with the a-contrario anomalies map

    loss = mean( e(i)*(input(i)-target(i)**2) )

    Parameters
    ----------
    radius: int
        Radius of the anomalies detection disk (in pixels)
    alpha: int
        Width of the anomalies detection ring (in pixel)

    """
    def __init__(self, radius=1, alpha=7):
        super().__init__()

        self.disk_mat = disk_patch(radius)
        self.ring_mat = ring_patch(radius, alpha)
        self.coefficient = math.sqrt(math.pi) * math.sqrt(1 - 1 / (alpha * alpha)) * radius
        self.sqrt2 = math.sqrt(2)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def forward(self, inputs, targets):
        """Calculate the loss

        Parameters
        ----------
        inputs: Tensor
            Network prediction
        targets: Tensor
            Expected result

        """
        # positive error map
        error_pos = inputs - targets
        inner_weights_pos = torch.Tensor(self.disk_mat).unsqueeze(0).unsqueeze(0).to(self.device)
        inner_weights_pos.require_grad = True
        outer_weights_pos = torch.Tensor(self.ring_mat).unsqueeze(0).unsqueeze(0).to(self.device)
        outer_weights_pos.require_grad = True

        inner_conv_pos = nn.Conv2d(1, 1, kernel_size=self.disk_mat.shape[0],
                                   stride=1,
                                   padding=int((self.disk_mat.shape[0] - 1) / 2),
                                   bias=False)
        outer_conv_pos = nn.Conv2d(1, 1, kernel_size=self.ring_mat.shape[0],
                                   stride=1,
                                   padding=int((self.ring_mat.shape[0] - 1) / 2),
                                   bias=False)
        with torch.no_grad():
            inner_conv_pos.weight = nn.Parameter(inner_weights_pos)
            outer_conv_pos.weight = nn.Parameter(outer_weights_pos)

        sigma_pos = torch.sqrt(torch.var(error_pos))
        stat_pos = self.coefficient * (
                    inner_conv_pos(error_pos) - outer_conv_pos(error_pos)) / sigma_pos

        stat_pos_norm = 0.5 * torch.erfc(stat_pos / self.sqrt2)

        # negative error map
        error_neg = targets - inputs
        inner_weights_neg = torch.Tensor(self.disk_mat).unsqueeze(0).unsqueeze(0).to(self.device)
        inner_weights_neg.require_grad = True
        outer_weights_neg = torch.Tensor(self.ring_mat).unsqueeze(0).unsqueeze(0).to(self.device)
        outer_weights_neg.require_grad = True

        inner_conv_neg = nn.Conv2d(1, 1, kernel_size=self.disk_mat.shape[0],
                                   stride=1,
                                   padding=int((self.disk_mat.shape[0] - 1) / 2),
                                   bias=False)
        outer_conv_neg = nn.Conv2d(1, 1, kernel_size=self.ring_mat.shape[0],
                                   stride=1,
                                   padding=int((self.ring_mat.shape[0] - 1) / 2),
                                   bias=False)
        with torch.no_grad():
            inner_conv_neg.weight = nn.Parameter(inner_weights_neg)
            outer_conv_neg.weight = nn.Parameter(outer_weights_neg)

        sigma_neg = torch.sqrt(torch.var(error_neg))
        stat_neg = self.coefficient * (
                    inner_conv_neg(error_neg) - outer_conv_neg(error_neg)) / sigma_neg

        stat_neg_norm = 0.5 * torch.erfc(stat_neg / self.sqrt2)

        # map combination
        th_map = (stat_pos_norm + stat_neg_norm)**2
        th_map = th_map/torch.sum(th_map)

        # MSE
        wmse = torch.mean(th_map*(inputs - targets) ** 2)
        return wmse
