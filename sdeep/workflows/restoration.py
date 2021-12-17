"""Restoration deep learning workflow"""
import os
from skimage.io import imsave
import torch

from .base import SWorkflow


class RestorationWorkflow(SWorkflow):
    def __init__(self, model, loss_fn, optimizer, train_data_loader,
                 val_data_loader, epochs=50):
        super().__init__(model, loss_fn, optimizer, train_data_loader,
                         val_data_loader, epochs)

    def after_train(self):
        """Instructions runs after the train."""
        SWorkflow.after_train(self)

        # create the output dir
        predictions_dir = os.path.join(self.out_dir, 'predictions')
        if os.path.isdir(self.out_dir):
            os.mkdir(predictions_dir)

        # predict on all the test set
        self.model.eval()
        with torch.no_grad():
            for x, y, names in self.val_data_loader:
                x, y = x.to(self.device), y.to(self.device)
                pred = self.model(x)
                for i in range(len(names)):
                    imsave(os.path.join(predictions_dir, names[i]), pred[i, :, :].cpu().numpy())
