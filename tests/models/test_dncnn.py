import torch
from sdeep.models.dncnn import DnCNN


def test_dncnn_receptive_field():
    model = DnCNN(num_of_layers=17, channels=1, features=64)
    assert model.receptive_field == 34
    assert model.input_shape == (34, 34)


def test_dncnn_gray():
    model = DnCNN(num_of_layers=17, channels=1, features=64)

    num_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    assert num_parameters == 556096

    dummy_input = torch.rand([1, 1, 34, 34])
    with torch.no_grad():
        output = model(dummy_input)
    assert output.shape[-1] == 34
    assert output.shape[-2] == 34


def test_dncnn_color():
    model = DnCNN(num_of_layers=17, channels=3, features=64)

    num_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    assert num_parameters == 558400

    dummy_input = torch.rand([1, 3, 128, 128])
    with torch.no_grad():
        output = model(dummy_input)
    assert output.shape[-1] == 128
    assert output.shape[-2] == 128
    assert output.shape[-3] == 3
