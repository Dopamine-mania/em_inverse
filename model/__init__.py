# 模型模块
from .model import (
    PyTorchDualBranchDeepONet,  # 原始双分支版本（保持兼容性）
    create_pytorch_dual_branch_deeponet  # 原始版本创建函数
)

from .enhanced_deeponet import (
    EnhancedPyTorchDualBranchDeepONet,     # 增强版双分支网络（推荐）
    create_enhanced_deeponet              # 增强版网络创建函数
)

__all__ = [
    # 原始网络（保持兼容性）
    'PyTorchDualBranchDeepONet',
    'create_pytorch_dual_branch_deeponet',

    # 增强网络（推荐使用）
    'EnhancedPyTorchDualBranchDeepONet',
    'create_enhanced_deeponet'
]