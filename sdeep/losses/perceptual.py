"""Implementation of perceptual loss using VGG"""
from typing import List

import torch
import torchvision


class VGGL1PerceptualLoss(torch.nn.Module):
    """VGG based perceptual loss

    :param weight: weight applied to the perceptual component
    :param resize: True to resize tensor to VGG input shape
    """
    def __init__(self, weight: float = 1.0, resize: bool = False):
        super().__init__()
        self.weight = weight
        blocks = []
        blocks.append(torchvision.models.vgg16(
            pretrained=True).features[:4].eval())
        blocks.append(torchvision.models.vgg16(
            pretrained=True).features[4:9].eval())
        blocks.append(torchvision.models.vgg16(
            pretrained=True).features[9:16].eval())
        blocks.append(torchvision.models.vgg16(
            pretrained=True).features[16:23].eval())
        for bl in blocks:
            for p in bl.parameters():
                p.requires_grad = False
        self.blocks = torch.nn.ModuleList(blocks)
        self.transform = torch.nn.functional.interpolate
        self.resize = resize
        self.register_buffer("mean", torch.tensor(
            [0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(
            [0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self.mae = torch.nn.L1Loss()

    def forward(self,
                source: torch.Tensor,
                target: torch.Tensor,
                feature_layers: List[int] = (0, 1, 2, 3),
                style_layers: List[int] = ()):
        """Torch forward method"""
        if source.shape[1] != 3:
            source = source.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)
        source = (source-self.mean) / self.std
        target = (target-self.mean) / self.std
        if self.resize:
            source = self.transform(source, mode='bilinear',
                                   size=(224, 224), align_corners=False)
            target = self.transform(target, mode='bilinear',
                                    size=(224, 224), align_corners=False)
        loss = 0.0
        x = source
        y = target
        for i, block in enumerate(self.blocks):
            x = block(x)
            y = block(y)
            if i in feature_layers:
                loss += torch.nn.functional.l1_loss(x, y)
            if i in style_layers:
                act_x = x.reshape(x.shape[0], x.shape[1], -1)
                act_y = y.reshape(y.shape[0], y.shape[1], -1)
                gram_x = act_x @ act_x.permute(0, 2, 1)
                gram_y = act_y @ act_y.permute(0, 2, 1)
                loss += torch.nn.functional.l1_loss(gram_x, gram_y)

        return self.mae(source, target) + self.weight*loss


export = [VGGL1PerceptualLoss]
