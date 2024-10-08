"""Evaluation tools for data classification"""
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
from ..interfaces import SEval


class EvalClassification(SEval):
    """Generate an evaluation of classification results"""
    def __init__(self):
        self.__y_true = None
        self.__y_pred = None

    def clear(self):
        self.__y_true = []
        self.__y_pred = []

    def eval_step(self,
                  prediction: torch.Tensor,
                  reference: torch.Tensor,
                  idx: str,
                  output_dir: Path = None):

        self.__y_true.append(reference.detach().cpu().item())
        self.__y_pred.append(torch.argmax(prediction).detach().cpu().item())

    def eval(self, output_dir: Path):
        """Run the global evaluation using results gathered by 'eval_step'

        Results must be saved as files in the 'output_dir'

        :param output_dir: Directory where the evaluation results are saved
        """
        # precision, recall, f1 scores
        f1_value = f1_score(self.__y_true, self.__y_pred, average=None)
        precision_value = precision_score(self.__y_true,
                                          self.__y_pred,
                                          average=None,
                                          zero_division=np.nan)
        recall_value = recall_score(self.__y_true,
                                    self.__y_pred,
                                    average=None,
                                    zero_division=np.nan)

        content = "Class,Precision,Recall,F1\n"
        for c, f1_val in enumerate(f1_value):
            content += f"{c},{precision_value[c]},{recall_value[c]},{f1_val}\n"

        with open(output_dir / "scores.csv", "w", encoding='utf-8') as file:
            file.write(content)

        # Confusion matrix
        c_mat = confusion_matrix(self.__y_true, self.__y_pred)
        pd.DataFrame(data=c_mat).to_csv(output_dir / "confusion_matrix.csv")


export = [EvalClassification]
