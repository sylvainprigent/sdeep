"""Implement an autoencoder"""
import torch
from torch import nn


class AEConvBlock(nn.Module):
    """Convolution block for UNet architecture

    This block is 2 convolution layers with a ReLU.
    An optional batch norm can be added after each convolution layer

    :param n_channels_in: Number of input channels (or features)
    :param n_channels_out: Number of output channels (or features)
    :param use_batch_norm: True to use the batch norm layers
    """
    def __init__(self,
                 n_channels_in: int,
                 n_channels_out: int,
                 num_layers: int = 2,
                 use_batch_norm: bool = True):
        super().__init__()

        self.layer = nn.Sequential()
        for i in range(num_layers):

            n_in = n_channels_in if i == 0 else n_channels_out
            self.layer.append(nn.Conv2d(n_in, n_channels_out,
                                        kernel_size=3, padding=1))
            if use_batch_norm:
                self.layer.append(nn.BatchNorm2d(n_channels_out))
            self.layer.append(nn.ReLU())

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Apply model

        :param inputs: Data to process
        :return: The data processed by the model
        """
        return self.layer(inputs)


class Autoencoder(nn.Module):
    """Implementation of image autoencoder network

    :param n_channels_in: Number of input channels,
    :param n_channels_out: Number of output channels,
    :param blocs_channels: Number of channels for each encoding/decoding blocs,
    :param image_size: Size of the image to process (needed when using token),
    :param use_token: True to add a fully connected embedding layer in the latent space,
    :param layer_per_bloc: Number of convolution layer per bloc
    """
    def __init__(self,
                 n_channels_in: int = 3,
                 n_channels_out: int = 3,
                 blocs_channels: list[int] = (32, 64, 128, 256, 512),
                 image_size: int = 128,
                 use_token: bool = False,
                 layer_per_bloc: int = 2):
        super().__init__()
        self.use_batch_norm = True
        self.receptive_field = image_size
        self.use_token = use_token

        encode_patch = image_size / pow(2, len(blocs_channels))
        encode_size = int(blocs_channels[-1] * encode_patch * encode_patch)
        print('encode_size=', encode_size)
        print(f'encoding reshape ({blocs_channels[-1]}, {encode_patch}, '
              f'{encode_patch}) to {encode_size}')

        self.encoder = nn.Sequential()
        for idx, bloc_channel in enumerate(blocs_channels):
            n_in = n_channels_in if idx == 0 else blocs_channels[idx-1]
            n_out = bloc_channel
            self.encoder.append(AEConvBlock(n_in, n_out, layer_per_bloc, self.use_batch_norm))
            self.encoder.append(nn.MaxPool2d((2, 2)))

        if self.use_token:
            self.token_encode = nn.Flatten()
            self.token = nn.Linear(encode_size, encode_size, bias=False)
        else:
            self.m_body = nn.Sequential(
                nn.Conv2d(blocs_channels[-1], blocs_channels[-1], 3, stride=1,
                          padding=1, bias=False)
            )

        self.decoder = nn.Sequential()
        for idx in reversed(range(len(blocs_channels))):
            n_in = blocs_channels[idx]
            n_out = n_channels_out if idx == 0 else blocs_channels[idx-1]
            self.decoder.append(nn.Upsample(scale_factor=2, mode='nearest'))
            self.decoder.append(AEConvBlock(n_in, n_out, layer_per_bloc, self.use_batch_norm))

        self.m_tail = nn.Conv2d(n_channels_out, n_channels_out, 3, stride=1, padding=1,
                                bias=False)

    def forward(self, x0: torch.Tensor) -> torch.Tensor:
        """torch forward method

        :param x0: Tensor to process
        :return: the processed tensor
        """
        #print("x0.shape=", x0.shape)
        x_encode = self.encoder(x0)
        #print("x_encode.shape=", x_encode.shape)

        if self.use_token:
            token = self.token_encode(x_encode)
            token = self.token(token)
            #token = self.token_decode(token)
            x = torch.reshape(token, x_encode.shape)
        else:
            x = self.m_body(x_encode)

        #print("x_body.shape=", x.shape)
        x_decode = self.decoder(x)
        #print("x_decode.shape=", x_decode.shape)
        x = self.m_tail(x_decode)
        #print("x_out.shape=", x.shape)
        return x

    def encode(self, x0: torch.Tensor) -> torch.Tensor:
        """Run only the encoder part of the autoencoder

        :param x0: Tensor to process
        :return: the data embedding
        """
        x_encode = self.encoder(x0)
        if self.use_token:
            token = self.token_encode(x_encode)
            x = self.token(token)
        else:
            x = self.m_body(x_encode)
        return x


export = [Autoencoder]
