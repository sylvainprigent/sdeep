import torch
from sdeep.models.unet import UNet


def test_unet_receptive_field():
    model = UNet()
    assert model.receptive_field == 32
    assert model.input_shape == (32, 32)


def test_unet_gray():
    model = UNet(n_channels_in=1, n_channels_out=1, n_feature_first=32)

    num_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    assert num_parameters == 472257

    dummy_input = torch.rand([1, 1, 128, 128])
    with torch.no_grad():
        output = model(dummy_input)
    assert output.shape[-1] == 128
    assert output.shape[-2] == 128


def test_unet_color():
    model = UNet(n_channels_in=3, n_channels_out=1, n_feature_first=32)

    num_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    assert num_parameters == 472833

    dummy_input = torch.rand([1, 3, 128, 128])
    with torch.no_grad():
        output = model(dummy_input)
    assert output.shape[-1] == 128
    assert output.shape[-2] == 128


def test_unet_multi():
    model = UNet(n_channels_in=1, n_channels_out=3, n_feature_first=32)

    num_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    assert num_parameters == 472323

    dummy_input = torch.rand([1, 1, 128, 128])
    with torch.no_grad():
        output = model(dummy_input)
    assert output.shape[-1] == 128
    assert output.shape[-2] == 128
    assert output.shape[-3] == 3
