# 损失函数模块
from .loss import CustomDeepONetLoss, PyTorchLossFunction, initialize_global_loss_fn, loss_wrapper

__all__ = ['CustomDeepONetLoss', 'PyTorchLossFunction', 'initialize_global_loss_fn', 'loss_wrapper']