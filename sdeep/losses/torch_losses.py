"""export pytorch losses for factory"""
import torch

export = [torch.nn.MSELoss,
          torch.nn.L1Loss,
          torch.nn.BCELoss,
          torch.nn.BCEWithLogitsLoss,
          torch.nn.CrossEntropyLoss]
