"""Implementation of the DnCNN network in pytorch"""
from torch import Tensor
from torch import nn


class DnCNN(nn.Module):
    """Implementation of the DnCNN network

    :param num_of_layers: Number of layers in the model
    :param channels: Number of channels in the images
    :param features: Number of features in hidden layers
    """
    def __init__(self,
                 num_of_layers: int = 17,
                 channels: int = 1,
                 features: int = 64):
        super().__init__()

        self.receptive_field = 2*num_of_layers
        self.input_shape = (self.receptive_field, self.receptive_field)
        kernel_size = 3
        padding = 1
        layers = [nn.Conv2d(in_channels=channels, out_channels=features,
                            kernel_size=kernel_size, padding=padding,
                            bias=True),
                  nn.ReLU(inplace=True)]
        for _ in range(num_of_layers - 2):
            layers.append(
                nn.Conv2d(in_channels=features, out_channels=features,
                          kernel_size=kernel_size, padding=padding,
                          bias=False))
            layers.append(nn.BatchNorm2d(features, eps=0.0001, momentum=0.95))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(in_channels=features, out_channels=channels,
                                kernel_size=kernel_size, padding=padding,
                                bias=False))
        self.dncnn = nn.Sequential(*layers)

    def forward(self, x: Tensor):
        """Network forward method

        :param x: Network input batch
        :return: Tensor containing the network output

        """
        y = x
        residue = self.dncnn(x)
        return y - residue


export = [DnCNN]
