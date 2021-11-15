"""UNet pytorch module

Implementation of the UNet network in pytorch

Classes
-------
DnCNN

"""

import torch
from torch import nn


class UNetConvBlock(nn.Module):
    """Convolution block for UNet architecture

    This block is 2 convolution layers with a ReLU.
    An optional batch norm can be added after each convolution layer

    Parameters
    ----------
    n_channels_in: int
        Number of input channels (or features)
    n_channels_out: int
        Number of output channels (or features)
    use_batch_norm: bool
        True to use the batch norm layers
    """
    def __init__(self, n_channels_in, n_channels_out, use_batch_norm=True):
        super().__init__()

        self.use_batch_norm = use_batch_norm
        self.conv1 = nn.Conv2d(n_channels_in, n_channels_out, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(n_channels_out)

        self.conv2 = nn.Conv2d(n_channels_out, n_channels_out, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(n_channels_out)

        self.relu = nn.ReLU()

    def forward(self, inputs):
        x = self.conv1(inputs)
        if self.use_batch_norm:
            x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        if self.use_batch_norm:
            x = self.bn2(x)
        x = self.relu(x)

        return x


class UNetEncoderBlock(nn.Module):
    """Encoder block of the UNet architecture

    The encoder block is a convolution block and a max polling layer

    Parameters
    ----------
    n_channels_in: int
        Number of input channels (or features)
    n_channels_out: int
        Number of output channels (or features)
    use_batch_norm: bool
        True to use the batch norm layers
    """
    def __init__(self, n_channels_in, n_channels_out, use_batch_norm=True):
        super().__init__()

        self.conv = UNetConvBlock(n_channels_in, n_channels_out, use_batch_norm)
        self.pool = nn.MaxPool2d((2, 2))

    def forward(self, inputs):
        x = self.conv(inputs)
        p = self.pool(x)

        return x, p


class UNetDecoderBlock(nn.Module):
    """Decoder block of a UNet architecture

    The decoder is a up-sampling concatenation and convolution block

    Parameters
    ----------
    n_channels_in: int
        Number of input channels (or features)
    n_channels_out: int
        Number of output channels (or features)
    use_batch_norm: bool
        True to use the batch norm layers
    """
    def __init__(self, n_channels_in, n_channels_out, use_batch_norm=True):
        super().__init__()

        self.up = nn.ConvTranspose2d(n_channels_in, n_channels_out, kernel_size=2, stride=2, padding=0)
        self.conv = UNetConvBlock(n_channels_out+n_channels_out, n_channels_out, use_batch_norm)

    def forward(self, inputs, skip):
        x = self.up(inputs)
        x = torch.cat([x, skip], axis=1)
        x = self.conv(x)

        return x


class UNet(nn.Module):
    """Implementation of the UNet network

    Parameters
    ----------
    n_channels_in: int
        Number of input channels (or features)
    n_channels_out: int
        Number of output channels (or features)
    n_feature_first: int
        Number of channels (or features) in the first convolution block
    use_batch_norm: bool
        True to use the batch norm layers

    """

    def __init__(self, n_channels_in=1, n_channels_out=1, n_feature_first=32, use_batch_norm=False):
        super().__init__()

        """ Encoder """
        self.e1 = UNetEncoderBlock(n_channels_in, n_feature_first, use_batch_norm)
        self.e2 = UNetEncoderBlock(n_feature_first, 2*n_feature_first, use_batch_norm)

        """ Bottleneck """
        self.b = UNetConvBlock(2*n_feature_first, 4*n_feature_first, use_batch_norm)

        """ Decoder """
        self.d1 = UNetDecoderBlock(4*n_feature_first, 2*n_feature_first, use_batch_norm)
        self.d2 = UNetDecoderBlock(2*n_feature_first, n_feature_first, use_batch_norm)

        """ Classifier """
        self.outputs = nn.Conv2d(n_feature_first, n_channels_out, kernel_size=1, padding=0)

    def forward(self, inputs):
        y = inputs
        """ Encoder """
        s1, p1 = self.e1(inputs)
        s2, p2 = self.e2(p1)

        """ Bottleneck """
        b = self.b(p2)

        """ Decoder """
        d1 = self.d1(b, s2)
        d2 = self.d2(d1, s1)

        """ Classifier """
        outputs = self.outputs(d2)

        return y + outputs
