# -*- coding: utf-8 -*-
import torch
import torch.nn as nn


class ResBlock(nn.Module):
    def __init__(self, in_channels=64, out_channels=64):
        super().__init__()
        # Mini DnCNN
        self.res = nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=1, bias=False),
                                 nn.ReLU(inplace=True),
                                 nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=1, bias=False))

    def forward(self, x):
        return x + self.res(x)


class DRUNetUp(nn.Module):
    """Implementation of the DRUNet network

    Parameters
    ----------
    in_nc: int
        Number of input channels
    out_nc: int
        Number of output channels
    nc: list
        Number of channels for each level of the UNet
    nb: int
        Number of residual blocs

    """
    def __init__(self, in_nc=1, out_nc=1, nc=[64, 128, 256, 512], nb=4):
        super().__init__()
        self.receptive_field = 128

        self.m_up0 = nn.ConvTranspose2d(in_nc, in_nc, kernel_size=2, stride=2, padding=0,
                                        bias=False)

        self.m_head = nn.Conv2d(in_nc, nc[0], 3, stride=1, padding=1, bias=False)

        self.m_down1 = nn.Sequential(
            *[ResBlock(nc[0], nc[0]) for _ in range(nb)],
            nn.Conv2d(nc[0], nc[1], kernel_size=2, stride=2, padding=0, bias=False),
        )

        self.m_down2 = nn.Sequential(
            *[ResBlock(nc[1], nc[1]) for _ in range(nb)],
            nn.Conv2d(nc[1], nc[2], kernel_size=2, stride=2, padding=0, bias=False),
        )

        self.m_down3 = nn.Sequential(
            *[ResBlock(nc[2], nc[2]) for _ in range(nb)],
            nn.Conv2d(nc[2], nc[3], kernel_size=2, stride=2, padding=0, bias=False),
        )

        self.m_body = nn.Sequential(*[ResBlock(nc[3], nc[3]) for _ in range(nb)])

        self.m_up3 = nn.Sequential(
            nn.ConvTranspose2d(nc[3], nc[2], kernel_size=2, stride=2, padding=0, bias=False),
            *[ResBlock(nc[2], nc[2]) for _ in range(nb)]
        )

        self.m_up2 = nn.Sequential(
            nn.ConvTranspose2d(nc[2], nc[1], kernel_size=2, stride=2, padding=0, bias=False),
            *[ResBlock(nc[1], nc[1]) for _ in range(nb)]
        )

        self.m_up1 = nn.Sequential(
            nn.ConvTranspose2d(nc[1], nc[0], kernel_size=2, stride=2, padding=0, bias=False),
            *[ResBlock(nc[0], nc[0]) for _ in range(nb)]
        )

        self.m_tail = nn.Conv2d(nc[0], out_nc, 3, stride=1, padding=1, bias=False)

        self.m_down_end = nn.Conv2d(out_nc, out_nc, kernel_size=2, stride=2, padding=0, bias=False)

    def forward(self, x0):
        x00 = self.m_up0(x0)
        x1 = self.m_head(x00)
        x2 = self.m_down1(x1)
        x3 = self.m_down2(x2)
        x4 = self.m_down3(x3)
        x = self.m_body(x4)
        x = self.m_up3(x + x4)
        x = self.m_up2(x + x3)
        x = self.m_up1(x + x2)
        x = self.m_tail(x + x1)
        x = self.m_down_end(x)
        return x


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    m = DRUNetUp()
    m.to(device)

    y = torch.randn(3, 1, 256, 256).to(device)
    x = m(y)
    print(x.shape)
