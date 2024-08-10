import torch


class ClassifierResBlock(torch.nn.Module):
    def __init__(self,
                 n_channels: int = 64,
                 n_blocs: int = 2,
                 use_batch_norm: bool = True
                 ):
        super().__init__()

        print('create ClassifierResBlock n_channels_in=', n_channels, ', n_blocs', n_blocs)

        self._seq = torch.nn.Sequential()
        for bloc in range(n_blocs):
            self._seq.append(torch.nn.Conv2d(n_channels, n_channels, kernel_size=3, padding=1))
            if use_batch_norm:
                self._seq.append(torch.nn.BatchNorm2d(n_channels))
            self._seq.append(torch.nn.ReLU())  # remove relu for last block ?

    def forward(self, x):
        return self._seq(x) - x


class Classifier(torch.nn.Module):
    """Image classifier Network"""
    def __init__(self,
                 image_size: int = 128,
                 n_channels_in: int = 3,
                 n_conv_per_res_bloc: int = 2,
                 n_res_per_bloc: int = 3,
                 blocs_channels: list[int] = (64, 128),
                 use_batch_norm: bool = True,
                 n_class: int = 2):
        super().__init__()

        encode_patch = image_size / pow(2, len(blocs_channels))
        encode_size = int(blocs_channels[-1] * encode_patch * encode_patch)

        self._features = []

        self._features.append(torch.nn.Conv2d(n_channels_in,
                                              blocs_channels[0],
                                              kernel_size=3,
                                              padding=1))

        for idx in range(len(blocs_channels)):
            nc_in = blocs_channels[idx]
            nc_out = blocs_channels[idx+1] if idx < len(blocs_channels)-1 else blocs_channels[idx]
            for _ in range(n_res_per_bloc):
                self._features.append(ClassifierResBlock(nc_in,
                                                         n_conv_per_res_bloc,
                                                         use_batch_norm))
            self._features.append(torch.nn.Conv2d(nc_in, nc_out,
                                                  kernel_size=2, stride=2, padding=0,
                                                  bias=False))

        self.flatten = torch.nn.Flatten()
        encode_size_out = 1024  # encode_size  # 1024
        self.fc1 = torch.nn.Linear(encode_size, encode_size_out, bias=False)
        self.fc2 = torch.nn.Linear(encode_size_out, n_class, bias=False)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        print('x input shape = ', x.shape)
        for feat in self._features:
            x = feat(x)
            print('x inner layer shape = ', x.shape)
        # x = self._features(x)
        print('features shape = ', x.shape)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return self.sigmoid(x)


export = [Classifier]
