"""UNet pytorch module

Implementation of the UNet network in pytorch

Classes
-------
CALayer
RCAB
ResidualGroup
RCAN

"""

from torch import nn
import math


class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


class RCAB(nn.Module):
    def __init__(
            self, n_feat, kernel_size, reduction,
            bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(RCAB, self).__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(nn.Conv2d(n_feat, n_feat, kernel_size,
                                          padding=(kernel_size//2), bias=bias))
            if bn:
                modules_body.append(nn.BatchNorm2d(n_feat))
            if i == 0:
                modules_body.append(act)
        modules_body.append(CALayer(n_feat, reduction))
        self.body = nn.Sequential(*modules_body)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x)
        res += x
        return res


class ResidualGroup(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, n_resblocks):
        super(ResidualGroup, self).__init__()
        modules_body = [
            RCAB(
                n_feat, kernel_size, reduction, bias=True, bn=False, act=nn.ReLU(True),
                res_scale=1) for _ in range(n_resblocks)]
        modules_body.append(nn.Conv2d(n_feat, n_feat, kernel_size, padding=(kernel_size // 2),
                                      bias=True))

        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res


class RCAN(nn.Module):
    """RCAN restoration network implementation

    Parameters
    ----------
    n_colors: int
        Number of colors in the input/output image
    n_resgroups: int
        Number of residual groups
    n_resblocks: int
        Number of residual blocs in each residual group
    n_feats: int
        Number of features
    reduction: int
        Reduction factor for channels downscaling
    scale: int
        Scale factor between the input and output image. Ex: scale 2 makes the output image twice
        the size of the input image

    """
    def __init__(self, n_colors=1, n_resgroups=10, n_resblocks=20, n_feats=64,
                 reduction=16, scale=1):
        super(RCAN, self).__init__()

        self.receptive_field = 64
        kernel_size = 3
        act = nn.ReLU(True)
        # define head module
        modules_head = [nn.Conv2d(n_colors, n_feats, kernel_size,
                                  padding=(kernel_size//2), bias=True)]
        # define body module
        modules_body = [
            ResidualGroup(
                n_feats, kernel_size, reduction,
                n_resblocks=n_resblocks) for _ in range(n_resgroups)]

        modules_body.append(nn.Conv2d(n_feats, n_feats, kernel_size,
                                      padding=(kernel_size//2), bias=True))
        # define tail module
        modules_tail = [
            UpSampler(scale, n_feats, act=False),
            nn.Conv2d(n_feats, n_colors, kernel_size, padding=(kernel_size//2), bias=True)]

        self.head = nn.Sequential(*modules_head)
        self.body = nn.Sequential(*modules_body)
        self.tail = nn.Sequential(*modules_tail)

    def forward(self, x):
        x = self.head(x)
        res = self.body(x)
        res += x
        x = self.tail(res)
        return x


class UpSampler(nn.Sequential):
    def __init__(self, scale, n_feat, bn=False, act=False, bias=True):

        m = []
        if (scale & (scale - 1)) == 0:    # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(nn.conv2(n_feat, 4 * n_feat, 3, padding=1, bias=bias))
                m.append(nn.PixelShuffle(2))
                if bn:
                    m.append(nn.BatchNorm2d(n_feat))
                if act:
                    m.append(act())
        elif scale == 3:
            m.append(nn.conv2(n_feat, 9 * n_feat, 3, padding=1, bias=bias))
            m.append(nn.PixelShuffle(3))
            if bn:
                m.append(nn.BatchNorm2d(n_feat))
            if act:
                m.append(act())
        else:
            raise NotImplementedError

        super(UpSampler, self).__init__(*m)
