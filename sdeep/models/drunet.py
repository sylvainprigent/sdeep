# -*- coding: utf-8 -*-
"""
Created on Wed Oct 20 09:45:12 2021

@author: sherbret
"""
from typing import List

import torch
from torch import nn


class ResBlock(nn.Module):
    """Residual block of DRUNet

    :param in_channels: Number of input channels
    :param out_channels: Number of output channels
    """
    def __init__(self, in_channels=64, out_channels=64):
        super().__init__()
        # Mini DnCNN
        self.res = nn.Sequential(nn.Conv2d(in_channels, out_channels, 3,
                                           stride=1, padding=1, bias=False),
                                 nn.ReLU(inplace=True),
                                 nn.Conv2d(in_channels, out_channels, 3,
                                           stride=1, padding=1, bias=False))

    def forward(self, x: torch.Tensor):
        """torch forward method

        :param x: Tensor to process
        """
        return x + self.res(x)


class DRUNet(nn.Module):
    """Implementation of the DRUNet network

    :param in_nc: Number of input channels
    :param out_nc: Number of output channels
    :param nc: Number of channels for each level of the UNet
    :param nb: Number of residual blocs
    """
    def __init__(self,
                 in_nc: int = 1,
                 out_nc: int = 1,
                 nc: List[int] = (64, 128, 256, 512),
                 nb: int = 4):
        super().__init__()
        self.receptive_field = 128
        self.input_shape = (128, 128)

        if len(nc) != 4:
            raise ValueError('nc must be of size 4')

        self.m_head = nn.Conv2d(in_nc, nc[0], 3, stride=1, padding=1,
                                bias=False)

        self.m_down1 = nn.Sequential(
            *[ResBlock(nc[0], nc[0]) for _ in range(nb)],
            nn.Conv2d(nc[0], nc[1], kernel_size=2, stride=2, padding=0,
                      bias=False),
        )

        self.m_down2 = nn.Sequential(
            *[ResBlock(nc[1], nc[1]) for _ in range(nb)],
            nn.Conv2d(nc[1], nc[2], kernel_size=2, stride=2, padding=0,
                      bias=False),
        )

        self.m_down3 = nn.Sequential(
            *[ResBlock(nc[2], nc[2]) for _ in range(nb)],
            nn.Conv2d(nc[2], nc[3], kernel_size=2, stride=2, padding=0,
                      bias=False),
        )

        self.m_body = nn.Sequential(*[ResBlock(nc[3],
                                               nc[3]) for _ in range(nb)])

        self.m_up3 = nn.Sequential(
            nn.ConvTranspose2d(nc[3], nc[2], kernel_size=2, stride=2,
                               padding=0, bias=False),
            *[ResBlock(nc[2], nc[2]) for _ in range(nb)]
        )

        self.m_up2 = nn.Sequential(
            nn.ConvTranspose2d(nc[2], nc[1], kernel_size=2, stride=2,
                               padding=0, bias=False),
            *[ResBlock(nc[1], nc[1]) for _ in range(nb)]
        )

        self.m_up1 = nn.Sequential(
            nn.ConvTranspose2d(nc[1], nc[0], kernel_size=2, stride=2,
                               padding=0, bias=False),
            *[ResBlock(nc[0], nc[0]) for _ in range(nb)]
        )

        self.m_tail = nn.Conv2d(nc[0], out_nc, 3, stride=1, padding=1,
                                bias=False)

    def forward(self, x0):
        """torch forward method

        :param x: Tensor to process
        """
        x1 = self.m_head(x0)
        x2 = self.m_down1(x1)
        x3 = self.m_down2(x2)
        x4 = self.m_down3(x3)
        x = self.m_body(x4)
        x = self.m_up3(x + x4)
        x = self.m_up2(x + x3)
        x = self.m_up1(x + x2)
        x = self.m_tail(x + x1)
        return x


export = [DRUNet]
