from .error_maps import SAContrarioMSELoss
from .perceptual import VGGL1PerceptualLoss
from .fmse import FMSELoss
from .frc import FRCLoss

__all__ = ['VGGL1PerceptualLoss', 'SAContrarioMSELoss', 'FRCLoss', 'FMSELoss']
