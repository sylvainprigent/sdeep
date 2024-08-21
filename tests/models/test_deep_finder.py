import torch
from sdeep.models.deep_finder import DeepFinder


def test_unet_receptive_field():
    model = DeepFinder()
    assert model.receptive_field == 48
    assert model.input_shape == (1, 48, 48)


def test_deep_finder_gray():
    model = DeepFinder(n_channels_in=1, n_channels_out=1, n_feature_first=32)

    num_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    assert num_parameters == 367025

    dummy_input = torch.rand([1, 1, 128, 128])
    with torch.no_grad():
        output = model(dummy_input)
    assert output.shape[-1] == 128
    assert output.shape[-2] == 128


def test_deep_finder_color():
    model = DeepFinder(n_channels_in=3, n_channels_out=1, n_feature_first=32)

    num_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    assert num_parameters == 367601

    dummy_input = torch.rand([1, 3, 128, 128])
    with torch.no_grad():
        output = model(dummy_input)
    assert output.shape[-1] == 128
    assert output.shape[-2] == 128


def test_deep_finder_multi():
    model = DeepFinder(n_channels_in=1, n_channels_out=3, n_feature_first=32)

    num_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    assert num_parameters == 367091

    dummy_input = torch.rand([1, 1, 128, 128])
    with torch.no_grad():
        output = model(dummy_input)
    assert output.shape[-1] == 128
    assert output.shape[-2] == 128
    assert output.shape[-3] == 3
