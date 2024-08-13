import torch
from sdeep.models.mnist import MNistClassifier


def test_mnist_shapes():
    model = MNistClassifier()
    assert model.receptive_field == 9
    assert model.input_shape == (28, 28)


def test_mnist():
    model = MNistClassifier()

    num_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    assert num_parameters == 21840

    dummy_input = torch.rand([1, 1, 28, 28])
    with torch.no_grad():
        output = model(dummy_input)
    assert output.shape[-1] == 10
