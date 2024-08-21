"""Module to implement image denoising evaluation module"""
from pathlib import Path

import numpy as np
import torch

from skimage.io import imsave

from ..interfaces import SEval


class EvalRestoration(SEval):
    """Generate an evaluation of denoising results"""
    def __init__(self):
        self.__mse = None
        self.__mae = None

    def clear(self):
        self.__mse = []
        self.__mae = []

    def eval_step(self,
                  prediction: torch.Tensor,
                  reference: torch.Tensor,
                  idx: str,
                  output_dir: Path = None):

        self.__mse.append(torch.mean((prediction-reference)**2).detach().cpu().item())
        self.__mae.append(torch.mean(torch.abs(prediction - reference)).detach().cpu().item())

        imsave(output_dir / f"error_{idx}.tif", (prediction-reference).detach().cpu().numpy())
        imsave(output_dir / f"prediction_{idx}.tif", prediction.detach().cpu().numpy())
        imsave(output_dir / f"ref_{idx}.tif", reference.detach().cpu().numpy())

    def eval(self, output_dir: Path):
        """Run the global evaluation using results gathered by 'eval_step'

        Results must be saved as files in the 'output_dir'

        :param output_dir: Directory where the evaluation results are saved
        """
        mse = np.asarray(self.__mse)
        mae = np.asarray(self.__mae)

        with open(output_dir / "scores.txt", "w", encoding='utf-8') as file:
            file.write(f"mse: {np.mean(mse)}, std: {np.std(mse)}\n"
                       f"mae: {np.mean(mae)}, std: {np.std(mae)}")


export = [EvalRestoration]
