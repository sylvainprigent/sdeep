"""Implement a fully connected network for classification"""
import torch
import torch.nn.functional as F


class MNistClassifier(torch.nn.Module):
    """Basic classifier for the MNist dataset to test and demo the classification framework"""
    def __init__(self):
        super().__init__()
        self.receptive_field = 9
        self.input_shape = (28, 28)
        self.conv1 = torch.nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = torch.nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = torch.nn.Dropout2d()
        self.fc1 = torch.nn.Linear(320, 50)
        self.fc2 = torch.nn.Linear(50, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the model

        :param x: Data to process
        :return: The processed data
        """
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.softmax(x, dim=1)


export = [MNistClassifier]
