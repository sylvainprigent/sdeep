import torch
from sdeep.models.drunet import DRUNet


def test_drunet_receptive_field():
    model = DRUNet(in_nc=1, out_nc=1, nc=[64, 128, 256, 512], nb=4)
    assert model.receptive_field == 128
    assert model.input_shape == (128, 128)


def test_drunet_gray():
    model = DRUNet(in_nc=1, out_nc=1, nc=[64, 128, 256, 512], nb=4)

    num_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    assert num_parameters == 32638080

    dummy_input = torch.rand([1, 1, 128, 128])
    with torch.no_grad():
        output = model(dummy_input)
    assert output.shape[-1] == 128
    assert output.shape[-2] == 128


def test_drunet_color():
    model = DRUNet(in_nc=3, out_nc=3, nc=[64, 128, 256, 512], nb=4)

    num_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    assert num_parameters == 32640384

    dummy_input = torch.rand([1, 3, 128, 128])
    with torch.no_grad():
        output = model(dummy_input)
    assert output.shape[-1] == 128
    assert output.shape[-2] == 128
    assert output.shape[-3] == 3
