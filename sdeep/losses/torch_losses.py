"""export pytorch losses for factory"""
import torch

export = [torch.nn.MSELoss,
          torch.nn.BCELoss,
          torch.nn.BCEWithLogitsLoss,
          torch.nn.CrossEntropyLoss]
