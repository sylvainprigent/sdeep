"""Interface for Eval modules"""
from abc import ABC, abstractmethod
from pathlib import Path
import torch


class SEval(ABC):
    """Interface for the evaluation modules

    An evaluation module must implement three methods:

    1. **clear**: to clear all the results between two epoch or two distinct evaluations
    2. **eval_step**: to evaluate a pair (y_true, y_pred) of results. This method might store
                      results for the global evaluation
    3. **eval**: Generate all the global evaluation scores and files
    """

    @abstractmethod
    def clear(self):
        """Clear all the data in the evaluation class"""

    @abstractmethod
    def eval_step(self,
                  prediction: torch.Tensor,
                  reference: torch.Tensor,
                  idx: str,
                  output_dir: Path = None):
        """Evaluate the metric for one sample

        This method is called by the workflow each time the model predict one sample at evaluation
        steps.

        :param prediction: Model prediction on the sample,
        :param reference: Reference results (ground truth)
        :param idx: identifier of the data
        :param output_dir: Directory where the evaluation results are saved
        """

    @abstractmethod
    def eval(self, output_dir: Path):
        """Run the global evaluation using results gathered by 'eval_step'

        Results must be saved as files in the 'output_dir'

        :param output_dir: Directory where the evaluation results are saved
        """
