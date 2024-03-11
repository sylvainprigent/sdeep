"""Implementation of the DeepFinder UNet network in pytorch"""

import torch
from torch import nn


class DeepFinder(nn.Module):
    """Implementation of the DRUNet network

    :param n_channels_in: Number of input channels
    :param n_channels_out: Number of output channels
    :param n_feature_first: Number of channels the first layer
    :param use_sigmoid: Apply sigmoid on the output layer
    """
    def __init__(self,
                 n_channels_in: int = 1,
                 n_channels_out: int = 1,
                 n_feature_first: int = 32,
                 use_sigmoid: bool = False):
        super().__init__()
        self.receptive_field = 48
        self.__use_sigmoid = use_sigmoid

        n_feature_l1 = n_feature_first
        n_feature_l2 = int(3/2*n_feature_first)
        n_feature_l3= int(2 * n_feature_first)

        self.block1 = nn.Sequential(nn.Conv2d(n_channels_in, n_feature_l1, 3,
                                              stride=1, padding=1, bias=False),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(n_feature_l1, n_feature_l1, 3,
                                              stride=1, padding=1, bias=False),
                                    nn.ReLU(inplace=True)
                                    )

        self.block2 = nn.Sequential(nn.Conv2d(n_feature_l1, n_feature_l2, 3,
                                              stride=1, padding=1, bias=False),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(n_feature_l2, n_feature_l2, 3,
                                              stride=1, padding=1, bias=False),
                                    nn.ReLU(inplace=True)
                                    )

        self.block_bn = nn.Sequential(nn.Conv2d(n_feature_l2, n_feature_l3, 3,
                                                stride=1, padding=1,
                                                bias=False),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(n_feature_l3, n_feature_l3, 3,
                                              stride=1, padding=1, bias=False),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(n_feature_l3, n_feature_l3, 3,
                                              stride=1, padding=1, bias=False),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(n_feature_l3, n_feature_l3, 3,
                                              stride=1, padding=1, bias=False),
                                    nn.ReLU(inplace=True)
                                    )

        self.block_up1 = nn.Sequential(
            nn.ConvTranspose2d(n_feature_l3, n_feature_l3, kernel_size=2,
                               stride=2, padding=0),
            nn.Conv2d(n_feature_l3, n_feature_l3, 3, stride=1, padding=1,
                      bias=False),
            nn.ReLU(inplace=True)
        )

        self.block_up2 = nn.Sequential(
            nn.Conv2d(n_feature_l2+n_feature_l3, n_feature_l2, 3, stride=1,
                      padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_feature_l2, n_feature_l2, 3, stride=1, padding=1,
                      bias=False),
            nn.ReLU(inplace=True)
        )

        self.block_up3 = nn.Sequential(
            nn.ConvTranspose2d(n_feature_l2, n_feature_l2, kernel_size=2,
                               stride=2, padding=0),
            nn.Conv2d(n_feature_l2, n_feature_l2, 3, stride=1, padding=1,
                      bias=False),
            nn.ReLU(inplace=True)
        )

        self.block_up4 = nn.Sequential(
            nn.Conv2d(n_feature_l1+n_feature_l2, n_feature_l1, 3, stride=1,
                      padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_feature_l1, n_feature_l1, 3, stride=1, padding=1,
                      bias=False),
            nn.ReLU(inplace=True)
        )

        self.pool1 = nn.MaxPool2d((2, 2))
        self.pool2 = nn.MaxPool2d((2, 2))

        self.conv_last = nn.Conv2d(
            in_channels=n_feature_l1, out_channels=n_channels_out,
            kernel_size=1
        )

    def forward(self, inputs):
        x1 = self.block1(inputs)
        x = self.pool1(x1)

        x2 = self.block2(x)
        x = self.pool2(x2)

        x = self.block_bn(x)

        x = self.block_up1(x)

        x = torch.cat([x, x2], axis=1)
        x = self.block_up2(x)

        x = self.block_up3(x)

        x = torch.cat([x, x1], axis=1)
        x = self.block_up4(x)

        x = self.conv_last(x)
        if self.__use_sigmoid:
            return torch.sigmoid(x)
        return x


export = [DeepFinder]
