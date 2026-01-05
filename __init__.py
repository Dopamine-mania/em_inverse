# 电磁场重构DeepONet模块
from .config import Config
from .data import MaskedDeepONetDataset
from .model import create_masked_deeponet
from .loss import CustomDeepONetLoss
from .utils import *

__all__ = [
    'Config',
    'MaskedDeepONetDataset',
    'create_masked_deeponet',
    'CustomDeepONetLoss'
]