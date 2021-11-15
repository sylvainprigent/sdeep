"""DnCNN pytorch module

Implementation of the DnCNN network in pytorch

Classes
-------
DnCNN

"""

from torch import nn


class DnCNN(nn.Module):
    """Implementation of the DnCNN network

    Parameters
    ----------
    num_of_layers: int
        Number of layers in the model
    channels: int
        Number of channels in the images
    features: int
        Number of features in hidden layers

    """

    def __init__(self, num_of_layers=17, channels=1, features=64):
        super().__init__()

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

    def forward(self, x):
        """Network forward method

        Parameters
        ----------
        x: Tensor
            Network input batch

        Returns
        -------
        Tensor containing the network output

        """
        y = x
        residue = self.dncnn(x)
        return y - residue
