"""Classification of tensor data deep learning workflow"""
from pathlib import Path

import torch

from ..utils import device
from .base import SWorkflowBase


class ClassificationWorkflow(SWorkflowBase):
    """Workflow to train and predict a classification neural network"""

    def after_train(self):
        """Instructions runs after the train."""
        SWorkflowBase.after_train(self)

        # create the output dir
        predictions_dir = Path(self.out_dir, 'predictions')
        predictions_dir.mkdir(parents=True, exist_ok=True)

        # predict on all the test set
        self.model_torch.eval()

        table_data = "data,y_pred,y_true\n"
        for x, y, names in self.val_data_loader:
            x = x.to(device())
            with torch.no_grad():
                prediction = self.model_torch(x)
            for i, name in enumerate(names):
                p_value = torch.argmax(prediction[i, ...])

                table_data += (f"{name},{p_value.cpu().numpy().item()},"
                               f"{y[i,...].cpu().numpy().item()}\n")

        with open(predictions_dir / "prediction.csv", "w", encoding='utf-8') as file:
            file.write(table_data)


export = [ClassificationWorkflow]
