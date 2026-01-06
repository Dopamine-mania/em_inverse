"""
DeepONet三层损失函数实现
保持原有的探针对齐、场强差值和平滑正则损失结构
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict, List, Optional
import sys
from pathlib import Path

# 添加父目录到路径
sys.path.append(str(Path(__file__).parent.parent))

from config.config import Config
from loss.spectral_loss_gpu import SpectralLossGPU  # Day 2: 使用GPU可微分版本


class CustomDeepONetLoss:
    """自定义DeepONet损失函数类，处理mask传递"""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.current_masks = None  # 存储当前批次的mask

        # 保存grid_bounds用于Spectral Loss（关键修复：必须传递正确的物理边界）
        self.grid_bounds = tuple(cfg.data.grid_bounds)  # (x_min, x_max, y_min, y_max)

        # 初始化Spectral Loss (Day 2: GPU可微分版本)
        if cfg.physics.spectral_loss_weight > 0:
            self.spectral_loss_fn = SpectralLossGPU(
                grid_size=cfg.data.grid_resolution,
                threshold_ratio=1000.0,  # 客户固定值
                cache_masks=True  # 启用缓存提升性能
            )
            print(f"[CustomDeepONetLoss] Spectral Loss GPU版本已启用 (权重={cfg.physics.spectral_loss_weight})")
            print(f"  ✅ 可微分: 梯度能传回DeepONet")
            print(f"  ✅ GPU加速: 约100倍性能提升")
            print(f"  ✅ Grid Bounds: {self.grid_bounds} (物理坐标范围)")
        else:
            self.spectral_loss_fn = None
            print(f"[CustomDeepONetLoss] Spectral Loss已禁用")

    def set_masks(self, masks: List[np.ndarray]):
        """设置当前批次的mask"""
        self.current_masks = masks

    def __call__(self, y_pred: torch.Tensor, y_true: torch.Tensor, masks: List[np.ndarray],
                 lengths: torch.Tensor = None,
                 coords: torch.Tensor = None,
                 sample_ids: torch.Tensor = None) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        修正版的损失函数 - 支持排除padding位置 + Spectral Loss
        Args:
            y_pred: [batch_size, N, 2] 所有位置预测
            y_true: [batch_size, N, 2] 所有位置真实值
            masks: List[np.ndarray] 每个样本的探针位置mask
            lengths: torch.Tensor [batch_size] 每个样本的有效长度
            coords: torch.Tensor [batch_size, N, 3] 空间坐标 (用于Spectral Loss)
            sample_ids: torch.Tensor [batch_size] 样本ID (用于Spectral Loss缓存)
        Returns:
            总损失和各项损失组成的字典
        """
        if self.current_masks is None:
            raise ValueError("Masks not set. Call set_masks() before loss computation.")

        total_loss_batch = 0
        loss_components_batch = {'probe_loss': 0.0, 'field_loss': 0.0, 'correlation_loss': 0.0, 'smooth_loss': 0.0, 'spectral_loss': 0.0}

        batch_size = y_pred.shape[0]

        for i in range(batch_size):
            mask = self.current_masks[i]  # [N] 当前样本的mask
            y_pred_i = y_pred[i]         # [N, 2] 当前样本预测
            y_true_i = y_true[i]         # [N, 2] 当前样本真实值

            # 获取当前样本的有效长度
            valid_length = None
            if lengths is not None:
                valid_length = lengths[i].item()

            # 1. 探针对齐损失 - 探针位置不受padding影响
            probe_loss = self.compute_probe_loss(y_pred_i, y_true_i, mask)

            # 2. 整体场损失 - 只计算有效长度内的数据
            field_loss = self.compute_field_loss(y_pred_i, y_true_i, valid_length)

            # 3. 实部相关性损失 - 只计算有效长度内的数据
            correlation_loss = self.compute_real_part_correlation_loss(y_pred_i, y_true_i, valid_length)

            # 4. 平滑正则损失 - 只计算有效长度内的数据
            smooth_loss = self.compute_smooth_loss(y_pred_i, valid_length)

            # 加权组合（先不包含Spectral Loss，因为它是批次级别的）
            total_loss = (self.cfg.physics.probe_loss_weight * probe_loss +
                         self.cfg.physics.field_loss_weight * field_loss +
                         self.cfg.physics.correlation_loss_weight * correlation_loss +
                         self.cfg.physics.smoothness_loss_weight * smooth_loss)

            total_loss_batch += total_loss
            loss_components_batch['probe_loss'] += probe_loss.item()
            loss_components_batch['field_loss'] += field_loss.item()
            loss_components_batch['correlation_loss'] += correlation_loss.item()
            loss_components_batch['smooth_loss'] += smooth_loss.item()

        # 计算Spectral Loss (批次级别)
        spectral_loss = torch.tensor(0.0, device=y_pred.device)
        if self.spectral_loss_fn is not None and coords is not None:
            try:
                # 🔧 关键修复：传递grid_bounds确保正确的物理坐标映射
                spectral_loss = self.spectral_loss_fn(
                    y_pred, y_true, coords, sample_ids,
                    grid_bounds=self.grid_bounds  # ✅ 传递配置中的物理边界
                )
                total_loss_batch += self.cfg.physics.spectral_loss_weight * spectral_loss
                loss_components_batch['spectral_loss'] = spectral_loss.item()
            except Exception as e:
                print(f"[Warning] Spectral Loss计算失败: {e}")
                loss_components_batch['spectral_loss'] = 0.0
        else:
            loss_components_batch['spectral_loss'] = 0.0

        # 平均批次损失（Data Loss部分）
        total_loss_batch /= batch_size
        for key in ['probe_loss', 'field_loss', 'correlation_loss', 'smooth_loss']:
            loss_components_batch[key] /= batch_size

        return total_loss_batch, loss_components_batch

    def compute_probe_loss(self, y_pred: torch.Tensor, y_true: torch.Tensor, mask: np.ndarray) -> torch.Tensor:
        """
        探针对齐损失 - 直接计算，无需插值！
        Args:
            y_pred: [N, 2] 当前样本预测
            y_true: [N, 2] 当前样本真实值
            mask: [N] 探针位置标记
        """
        probe_pred = y_pred[mask]      # [50, 2] 探针位置预测
        probe_true = y_true[mask]      # [50, 2] 探针位置真实

        if self.cfg.physics.use_probe_weight:
            # 使用幅值加权
            probe_magnitude = torch.sqrt(probe_true[:, 0]**2 + probe_true[:, 1]**2)
            weights = 1.0 + self.cfg.physics.probe_alpha * probe_magnitude
            probe_loss = torch.mean(weights.unsqueeze(1) * (probe_pred - probe_true)**2)
        else:
            probe_loss = F.mse_loss(probe_pred, probe_true)

        return probe_loss

    def compute_field_loss(self, y_pred: torch.Tensor, y_true: torch.Tensor, valid_length: int = None) -> torch.Tensor:
        """
        场强差值损失 - 只计算有效位置，排除padding
        Args:
            y_pred: [N, 2] 预测场值
            y_true: [N, 2] 真实场值
            valid_length: int 有效长度（排除padding）
        """
        if valid_length is not None and valid_length < y_pred.shape[0]:
            # 只计算有效长度内的损失，排除padding
            y_pred_valid = y_pred[:valid_length]
            y_true_valid = y_true[:valid_length]
            return F.mse_loss(y_pred_valid, y_true_valid)
        else:
            # 如果没有提供valid_length，计算整个张量（向后兼容）
            return F.mse_loss(y_pred, y_true)

    def compute_real_part_correlation_loss(self, y_pred: torch.Tensor, y_true: torch.Tensor, valid_length: int = None) -> torch.Tensor:
        """
        实部相关性损失 - 基于有效位置点到y=x线的距离之和
        Args:
            y_pred: [N, 2] 预测场值，最后一维为[real, imag]
            y_true: [N, 2] 真实场值，最后一维为[real, imag]
            valid_length: int 有效长度（排除padding）
        Returns:
            有效位置所有点到y=x线的平均距离作为损失
        """
        # 只处理有效长度内的数据
        if valid_length is not None and valid_length < y_pred.shape[0]:
            y_pred_valid = y_pred[:valid_length]
            y_true_valid = y_true[:valid_length]
        else:
            y_pred_valid = y_pred
            y_true_valid = y_true

        pred_real = y_pred_valid[:, 0]  # [N_valid] 预测实部
        true_real = y_true_valid[:, 0]  # [N_valid] 真实实部

        # 计算每个点到y=x线的垂直距离
        # y=x线的标准形式: y - x = 0
        # 点(ax, ay)到线Ax+By+C=0的距离: |A*ax + B*ay + C| / sqrt(A^2 + B^2)
        # 对于y=x线: x - y = 0, 所以A=1, B=-1, C=0
        # 距离 = |pred_real - true_real| / sqrt(2)

        distances = torch.abs(pred_real - true_real) / torch.sqrt(torch.tensor(2.0, device=y_pred.device))

        # 改为平均距离，避免点数大的样本主导损失
        if len(distances) > 0:
            mean_distance = torch.mean(distances)
        else:
            mean_distance = torch.tensor(0.0, device=y_pred.device)

        return mean_distance

    def compute_smooth_loss(self, y_pred: torch.Tensor, valid_length: int = None) -> torch.Tensor:
        """
        平滑正则损失 - 只计算有效位置，排除padding
        Args:
            y_pred: [N, 2] 场值预测
            valid_length: int 有效长度（排除padding）
        """
        # 只处理有效长度内的数据
        if valid_length is not None and valid_length < y_pred.shape[0]:
            y_pred_valid = y_pred[:valid_length]
        else:
            y_pred_valid = y_pred

        if self.cfg.physics.smooth_type == "laplacian":
            return self.compute_laplacian_smoothness(y_pred_valid)
        elif self.cfg.physics.smooth_type == "TV":
            return self.compute_total_variation_smoothness(y_pred_valid)
        else:
            return torch.tensor(0.0, device=y_pred.device)

    def compute_laplacian_smoothness(self, y_pred: torch.Tensor) -> torch.Tensor:
        """计算拉普拉斯平滑正则损失"""
        real_part = y_pred[:, 0]  # 实部
        imag_part = y_pred[:, 1]  # 虚部

        # 简化版本的拉普拉斯计算
        # 这里假设数据是规则网格，实际应用中需要根据数据结构调整
        laplacian_real = self._compute_discrete_laplacian(real_part)
        laplacian_imag = self._compute_discrete_laplacian(imag_part)

        # 平滑正则损失
        smooth_loss = torch.mean(laplacian_real**2) + torch.mean(laplacian_imag**2)
        return smooth_loss

    def _compute_discrete_laplacian(self, field: torch.Tensor) -> torch.Tensor:
        """
        计算离散拉普拉斯算子
        这是一个简化实现，假设数据是规则网格
        """
        # 获取数据形状
        num_points = field.shape[0]

        # 如果点数太少，返回零
        if num_points < 9:
            return torch.zeros_like(field)

        # 简化版本：使用最近邻的近似
        # 注意：这不是真正的2D拉普拉斯，只是一个平滑性正则项
        laplacian = torch.zeros_like(field)

        # 使用滑动窗口计算近似拉普拉斯
        for i in range(1, num_points - 1):
            # 简单的3点近似
            laplacian[i] = field[i-1] - 2*field[i] + field[i+1]

        return laplacian

    def compute_total_variation_smoothness(self, y_pred: torch.Tensor) -> torch.Tensor:
        """计算全变分平滑损失"""
        real_part = y_pred[:, 0]  # 实部
        imag_part = y_pred[:, 1]  # 虚部

        # 计算梯度（简化版本）
        real_grad = torch.diff(real_part)
        imag_grad = torch.diff(imag_part)

        # TV损失
        tv_loss = torch.mean(torch.abs(real_grad)) + torch.mean(torch.abs(imag_grad))

        return tv_loss


class PyTorchLossFunction:
    """PyTorch原生损失函数（用于PyTorch模型训练）- 支持Spectral Loss"""

    def __init__(self, cfg: Config):
        self.cfg = cfg

        # 初始化Spectral Loss (Day 2核心：GPU加速 + 可微分)
        self.spectral_loss_fn = None
        if cfg.physics.spectral_loss_weight > 0:
            from loss.spectral_loss_gpu import SpectralLossGPU
            self.spectral_loss_fn = SpectralLossGPU(
                grid_size=cfg.data.grid_resolution,
                threshold_ratio=1000.0,
                cache_masks=True
            )
            print(f"[PyTorchLossFunction] 🚀 Spectral Loss GPU版本已启用 (weight={cfg.physics.spectral_loss_weight})")
            print(f"  ✅ 可微分: 梯度能传回DeepONet")
            print(f"  ✅ GPU加速: 约100倍性能提升")

    def __call__(self, y_pred: torch.Tensor, y_true: torch.Tensor,
                 masks: List[torch.Tensor], lengths: torch.Tensor = None,
                 coords: torch.Tensor = None, sample_ids: torch.Tensor = None) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        PyTorch损失函数 - 支持排除padding位置和Spectral Loss (Day 1核心)
        Args:
            y_pred: [batch_size, N, 2] 预测值
            y_true: [batch_size, N, 2] 真实值
            masks: [batch_size, N] 探针位置掩码
            lengths: torch.Tensor [batch_size] 每个样本的有效长度
            coords: torch.Tensor [batch_size, N, 3] 空间坐标，用于Spectral Loss
            sample_ids: torch.Tensor [batch_size] 样本ID，用于Spectral Loss mask缓存
        """
        batch_size = y_pred.shape[0]
        total_loss = 0
        loss_dict = {'probe_loss': 0.0, 'field_loss': 0.0, 'correlation_loss': 0.0, 'smooth_loss': 0.0, 'spectral_loss': 0.0}

        for i in range(batch_size):
            mask = masks[i].bool()  # 转换为布尔张量
            y_pred_i = y_pred[i]
            y_true_i = y_true[i]

            # 获取当前样本的有效长度
            valid_length = None
            if lengths is not None:
                valid_length = lengths[i].item()

            # 探针对齐损失 - 探针位置不受padding影响
            # mask是1D布尔张量，需要应用到2D预测的每一维
            probe_pred = y_pred_i[mask]  # 这会自动选择第0维的mask应用到第1维
            probe_true = y_true_i[mask]
            probe_loss = F.mse_loss(probe_pred, probe_true)

            # 场强差值损失 - 始终只计算有效长度内的数据，排除padding
            if valid_length is not None and valid_length > 0 and valid_length < y_pred_i.shape[0]:
                # 有明确的valid_length且小于总长度，截断计算
                field_loss = F.mse_loss(y_pred_i[:valid_length], y_true_i[:valid_length])
            elif valid_length is not None and valid_length > 0:
                # valid_length等于总长度，使用全部数据（此时没有padding）
                field_loss = F.mse_loss(y_pred_i, y_true_i)
            else:
                # 没有valid_length信息，降级为使用mask（探针mask可能不包含所有有效点）
                # 这不是一个好的解决方案，但作为fallback
                if torch.any(mask):
                    # 如果有探针mask，使用非探针点计算场损失
                    non_probe_mask = ~mask
                    if torch.any(non_probe_mask):
                        field_loss = F.mse_loss(y_pred_i[non_probe_mask], y_true_i[non_probe_mask])
                    else:
                        field_loss = torch.tensor(0.0, device=y_pred.device)
                else:
                    # 最后的fallback：计算全部（这会包含padding，但概率很低）
                    field_loss = F.mse_loss(y_pred_i, y_true_i)

            # 实部相关性损失 - 只计算有效长度内的数据
            correlation_loss = self._compute_real_part_correlation_loss_pt(y_pred_i, y_true_i, valid_length)

            # 平滑正则损失 - 只计算有效长度内的数据
            smooth_loss = self._compute_smooth_loss_pt(y_pred_i, valid_length)

            # 加权组合
            sample_loss = (self.cfg.physics.probe_loss_weight * probe_loss +
                          self.cfg.physics.field_loss_weight * field_loss +
                          self.cfg.physics.correlation_loss_weight * correlation_loss +
                          self.cfg.physics.smoothness_loss_weight * smooth_loss)

            total_loss += sample_loss
            loss_dict['probe_loss'] += probe_loss.item()
            loss_dict['field_loss'] += field_loss.item()
            loss_dict['correlation_loss'] += correlation_loss.item()
            loss_dict['smooth_loss'] += smooth_loss.item()

        # Day 1核心：计算Spectral Loss (批次级别，在循环外)
        spectral_loss = torch.tensor(0.0, device=y_pred.device)
        if self.spectral_loss_fn is not None and coords is not None:
            try:
                spectral_loss = self.spectral_loss_fn(y_pred, y_true, coords, sample_ids)
                total_loss += self.cfg.physics.spectral_loss_weight * spectral_loss
                loss_dict['spectral_loss'] = spectral_loss.item()
            except Exception as e:
                print(f"[WARNING] Spectral Loss计算失败: {e}")
                loss_dict['spectral_loss'] = 0.0

        # 平均
        total_loss /= batch_size
        for key in loss_dict:
            if key != 'spectral_loss':  # spectral_loss已经是批次平均的，不需要再除
                loss_dict[key] /= batch_size

        return total_loss, loss_dict

    def _compute_real_part_correlation_loss_pt(self, y_pred: torch.Tensor, y_true: torch.Tensor, valid_length: int = None) -> torch.Tensor:
        """
        PyTorch版本的实部相关性损失 - 基于有效位置点到y=x线的距离之和
        Args:
            y_pred: [N, 2] 预测场值，最后一维为[real, imag]
            y_true: [N, 2] 真实场值，最后一维为[real, imag]
            valid_length: int 有效长度（排除padding）
        Returns:
            有效位置所有点到y=x线的距离之和作为损失
        """
        # 只处理有效长度内的数据
        if valid_length is not None and valid_length < y_pred.shape[0]:
            y_pred_valid = y_pred[:valid_length]
            y_true_valid = y_true[:valid_length]
        else:
            y_pred_valid = y_pred
            y_true_valid = y_true

        pred_real = y_pred_valid[:, 0]  # [N_valid] 预测实部
        true_real = y_true_valid[:, 0]  # [N_valid] 真实实部

        # 计算每个点到y=x线的垂直距离
        # 距离 = |pred_real - true_real| / sqrt(2)
        distances = torch.abs(pred_real - true_real) / torch.sqrt(torch.tensor(2.0, device=y_pred.device))

        # 改为平均距离，避免点数大的样本主导损失
        if len(distances) > 0:
            mean_distance = torch.mean(distances)
        else:
            mean_distance = torch.tensor(0.0, device=y_pred.device)

        return mean_distance

    def _compute_smooth_loss_pt(self, y_pred: torch.Tensor, valid_length: int = None) -> torch.Tensor:
        """PyTorch版本的平滑损失计算 - 只计算有效位置，避免乱序点噪声"""
        # 只处理有效长度内的数据
        if valid_length is not None and valid_length < y_pred.shape[0]:
            y_pred_valid = y_pred[:valid_length]
        else:
            y_pred_valid = y_pred

        # 由于数据点是乱序的，传统的laplacian或TV差分会产生噪声
        # 这里改为简单的L2正则化，鼓励预测值趋于0（避免过拟合）
        if self.cfg.physics.smoothness_loss_weight == 0.0:
            return torch.tensor(0.0, device=y_pred.device)

        # 简单的L2正则化作为平滑项
        return torch.mean(y_pred_valid**2) * 0.01  # 很小的权重


# 全局损失函数实例（用于DeepXDE）
global_loss_fn = None


def loss_wrapper(y_pred, y_true):
    """DeepXDE兼容的损失函数包装器"""
    global global_loss_fn
    if global_loss_fn is None:
        raise ValueError("Global loss function not initialized")
    return global_loss_fn(y_pred, y_true)


def initialize_global_loss_fn(cfg: Config):
    """初始化全局损失函数"""
    global global_loss_fn
    global_loss_fn = CustomDeepONetLoss(cfg)
    return global_loss_fn


def test_loss_function():
    """测试损失函数"""
    cfg = Config()

    # 创建测试数据
    batch_size = 2
    num_points = 128
    num_probes = 50

    y_true = torch.randn(batch_size, num_points, 2)
    y_pred = y_true + 0.1 * torch.randn(batch_size, num_points, 2)

    # 创建掩码
    masks = []
    for i in range(batch_size):
        mask = np.zeros(num_points, dtype=bool)
        probe_indices = np.random.choice(num_points, num_probes, replace=False)
        mask[probe_indices] = True
        masks.append(mask)

    # 测试CustomDeepONetLoss
    print("测试CustomDeepONetLoss...")
    loss_fn = CustomDeepONetLoss(cfg)
    loss_fn.set_masks(masks)

    try:
        total_loss, loss_dict = loss_fn(y_pred, y_true)
        print(f"总损失: {total_loss.item():.6f}")
        print(f"损失组件: {loss_dict}")
        print("CustomDeepONetLoss测试成功！")
    except Exception as e:
        print(f"CustomDeepONetLoss测试失败: {e}")

    # 测试PyTorchLossFunction
    print("\n测试PyTorchLossFunction...")
    pytorch_loss_fn = PyTorchLossFunction(cfg)
    torch_masks = [torch.from_numpy(mask) for mask in masks]

    try:
        total_loss, loss_dict = pytorch_loss_fn(y_pred, y_true, torch_masks)
        print(f"总损失: {total_loss.item():.6f}")
        print(f"损失组件: {loss_dict}")
        print("PyTorchLossFunction测试成功！")
    except Exception as e:
        print(f"PyTorchLossFunction测试失败: {e}")


if __name__ == "__main__":
    test_loss_function()