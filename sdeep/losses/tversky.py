"""Implementation of the Tversky semantic segmentation loss"""
import torch


class TverskyLoss(torch.nn.Module):
    """Tversky loss implementation

    :param alpha: alpha weight
    :param beta: beta weight
    :param weights: classes weights
    """
    def __init__(self,
                 alpha: float = 0.5,
                 beta: float = 0.5,
                 weights: torch.Tensor = None):
        super().__init__()
        self.__alpha = alpha
        self.__beta = beta
        self.__weights = weights

    def forward(self, x, target):
        # per class Tversky
        ones = torch.ones(x.shape).to(self.__device)
        p_0 = target
        p_1 = ones - target
        g_0 = x
        g_1 = ones - x

        num = torch.sum(p_0 * g_0, (2, 3))
        den = (num + self.__alpha*torch.sum(p_0*g_1, (2, 3)) +
               self.__beta * torch.sum(p_1*g_0, (2, 3)))

        # aggregate classes
        t_value = torch.sum((num + 1e-6) / (den + 1e-6), 1)
        n_classes = x.shape[1]*torch.ones(x.shape[0]).to(self.__device)
        return torch.mean(n_classes - t_value)


export = [TverskyLoss]
