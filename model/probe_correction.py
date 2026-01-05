"""
探针硬穿透+RBF平滑插值校正模块
实现基于残差的连续场校正机制
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, Dict
import sys
from pathlib import Path

# 添加模块路径
sys.path.append(str(Path(__file__).parent.parent))

from config import Config


class ProbeCorrectionModule(nn.Module):
    """
    探针硬穿透+RBF平滑插值校正模块
    1. 基线预测 y_base
    2. 探针残差 r = y_probe_true - y_base_probe
    3. RBF平滑插值 delta(x) = Σ w_i(x) * r_i
    4. 最终输出 y_final = y_base + delta
    """

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.enable_correction = cfg.physics.enable_rbf_correction

        # RBF参数
        if cfg.physics.learn_rbf_gamma:
            # 可学习的RBF带宽参数，确保为正值
            self.rbf_gamma = nn.Parameter(torch.tensor(cfg.physics.rbf_gamma, dtype=torch.float32))
        else:
            # 固定的RBF带宽参数
            self.register_buffer('rbf_gamma', torch.tensor(cfg.physics.rbf_gamma, dtype=torch.float32))

        # 频率感知插值
        self.frequency_aware = cfg.physics.rbf_frequency_aware

        print(f"探针校正模块初始化:")
        print(f"   启用RBF校正: {'是' if self.enable_correction else '否'}")
        print(f"   RBF带宽γ: {self.rbf_gamma.item():.3f} ({'可学习' if cfg.physics.learn_rbf_gamma else '固定'})")
        print(f"   频率感知插值: {'是' if self.frequency_aware else '否'}")

    def extract_probe_coordinates(self, branch_data: torch.Tensor) -> torch.Tensor:
        """
        从branch数据中提取探针坐标
        Args:
            branch_data: [batch_size, probe_count * 5] 探针数据 (x,y,z,freq,measurement)
        Returns:
            probe_coords: [batch_size, probe_count, 2] 探针坐标 (x,y)
        """
        batch_size = branch_data.shape[0]
        probe_count = self.cfg.deeponet.probe_count

        # 重塑为 [batch_size, probe_count, 5]
        branch_reshaped = branch_data.view(batch_size, probe_count, 5)

        # 提取x,y坐标
        probe_coords = branch_reshaped[:, :, :2]  # [batch_size, probe_count, 2]

        return probe_coords

    def extract_trunk_coordinates(self, trunk_data: torch.Tensor) -> torch.Tensor:
        """
        从trunk数据中提取坐标
        Args:
            trunk_data: [batch_size, N, 3] or [batch_size, max_len, 3] (x,y,freq)
        Returns:
            trunk_coords: [batch_size, N, 2] or [batch_size, max_len, 2] (x,y)
        """
        # 提取x,y坐标
        trunk_coords = trunk_data[:, :, :2]  # [batch_size, N, 2]

        return trunk_coords

    def compute_rbf_weights(self, trunk_coords: torch.Tensor, probe_coords: torch.Tensor,
                           valid_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        计算Gaussian RBF权重
        Args:
            trunk_coords: [batch_size, N, 2] 场点坐标
            probe_coords: [batch_size, probe_count, 2] 探针坐标
            valid_mask: [batch_size, N] 有效位置mask (用于排除padding)
        Returns:
            rbf_weights: [batch_size, N, probe_count] 归一化RBF权重
        """
        batch_size, N, _ = trunk_coords.shape
        probe_count = probe_coords.shape[1]

        # 计算距离矩阵 [batch_size, N, probe_count]
        # 使用欧氏距离 ||x - x_probe||²
        diff = trunk_coords.unsqueeze(2) - probe_coords.unsqueeze(1)  # [batch_size, N, probe_count, 2]
        distances_sq = torch.sum(diff ** 2, dim=-1)  # [batch_size, N, probe_count]

        # 应用有效区域mask到距离（只处理非padding位置）
        if valid_mask is not None:
            # valid_mask: [batch_size, N] -> [batch_size, N, 1]
            mask_expanded = valid_mask.unsqueeze(2).float()
            # 对padding位置设置极大距离，使权重接近0
            distances_sq = distances_sq * mask_expanded + (1 - mask_expanded) * 1e6

        # Gaussian RBF权重 w_ij = exp(-γ * ||x_i - x_j||²)
        rbf_weights = torch.exp(-self.rbf_gamma * distances_sq)  # [batch_size, N, probe_count]

        # 归一化权重 Σ_j w_ij = 1
        weight_sums = torch.sum(rbf_weights, dim=2, keepdim=True)  # [batch_size, N, 1]

        # 避免除零，设置最小值1e-8
        weight_sums = torch.clamp(weight_sums, min=1e-8)

        rbf_weights = rbf_weights / weight_sums  # [batch_size, N, probe_count]

        # 注意：这里不再额外应用mask，因为RBF权重本身已经处理了padding

        return rbf_weights

    def forward(self, y_base: torch.Tensor, branch_real: torch.Tensor,
                trunk: torch.Tensor, mask: Optional[torch.Tensor] = None,
                lengths: Optional[torch.Tensor] = None,
                probe_true_values: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        前向传播：执行探针硬穿透+RBF平滑插值校正
        Args:
            y_base: [batch_size, N, 2] 基线预测结果
            branch_real: [batch_size, probe_count * 5] 探针数据
            trunk: [batch_size, N, 3] 场点坐标和频率
            mask: [batch_size, N] 探针位置mask
            lengths: [batch_size] 有效长度
            probe_true_values: [batch_size, probe_count, 2] 真实探针值（用于计算残差）
        Returns:
            y_final: [batch_size, N, 2] 最终预测结果
            correction_info: Dict 包含校正过程的中间信息
        """
        batch_size, N, _ = y_base.shape

        if not self.enable_correction:
            # 如果禁用校正，直接返回基线预测
            return y_base, {'y_base': y_base, 'delta': torch.zeros_like(y_base)}

        # 1. 提取坐标信息
        probe_coords = self.extract_probe_coordinates(branch_real)  # [batch_size, probe_count, 2]
        trunk_coords = self.extract_trunk_coordinates(trunk)        # [batch_size, N, 2]

        # 2. 在探针位置提取基线预测值
        y_base_probe_list = []

        for i in range(batch_size):
            if mask is not None:
                probe_mask = mask[i].bool()  # [N]
                # 确保y_base[i]和probe_mask的维度匹配
                y_base_i = y_base[i]  # [N, 2]
                y_base_probe_i = y_base_i[probe_mask]  # [num_probes, 2]

                # 如果提取的探针数量与配置不一致，进行适配
                if y_base_probe_i.shape[0] != probe_coords.shape[1]:
                    # 如果数量不匹配，截断或填充
                    target_count = probe_coords.shape[1]
                    if y_base_probe_i.shape[0] > target_count:
                        y_base_probe_i = y_base_probe_i[:target_count]
                    else:
                        # 填充零，确保维度匹配
                        padding = torch.zeros(target_count - y_base_probe_i.shape[0], 2, device=y_base.device)
                        y_base_probe_i = torch.cat([y_base_probe_i, padding], dim=0)
            else:
                # 如果没有mask，假设前probe_count个点是探针（临时方案）
                probe_count = probe_coords.shape[1]
                y_base_probe_i = y_base[i, :probe_count]  # [probe_count, 2]

            y_base_probe_list.append(y_base_probe_i)

        y_base_probe = torch.stack(y_base_probe_list, dim=0)  # [batch_size, probe_count, 2]

        # 3. 计算探针残差 r = y_probe_true - y_base_probe
        if probe_true_values is not None:
            # 验证探针对齐情况
            expected_probe_count = y_base_probe.shape[1]
            actual_probe_count = probe_true_values.shape[1]

            if actual_probe_count != expected_probe_count:
                # 检查是否启用警告（通过检查配置中的标志）
                enable_warning = hasattr(self.cfg, 'physics') and self.cfg.physics.probe_alignment_warning
                if enable_warning:
                    print(f"⚠️ 探针校正模块警告: 探针真值与基线预测数量不匹配")
                    print(f"   真值探针数: {actual_probe_count}")
                    print(f"   基线探针数: {expected_probe_count}")

                if actual_probe_count > expected_probe_count:
                    if enable_warning:
                        print(f"   截断真值探针到 {expected_probe_count} 个")
                    probe_true_values = probe_true_values[:, :expected_probe_count, :]
                else:
                    if enable_warning:
                        print(f"   填充真值探针到 {expected_probe_count} 个")
                    padding = torch.zeros(batch_size, expected_probe_count - actual_probe_count, 2,
                                         device=y_base_probe.device)
                    probe_true_values = torch.cat([probe_true_values, padding], dim=1)

            # 确保所有张量在相同设备上（移动到基线预测设备）
            probe_true_values = probe_true_values.to(y_base_probe.device)
            probe_residuals = probe_true_values - y_base_probe  # [batch_size, probe_count, 2]
        else:
            # 如果没有真实探针值，假设残差为0（基线预测即是最终预测）
            probe_residuals = torch.zeros_like(y_base_probe)

        # 4. 计算RBF权重
        # 构造有效区域mask，只用于排除padding位置，不使用探针mask
        valid_mask = None
        if lengths is not None:
            max_len = trunk.size(1)
            idx = torch.arange(max_len, device=trunk.device).unsqueeze(0)  # [1, max_len]
            # 确保lengths与idx在相同设备上
            lengths = lengths.to(trunk.device)
            valid_mask = idx < lengths.unsqueeze(1)  # [batch_size, max_len]
            # 截断到实际长度
            valid_mask = valid_mask[:, :trunk.size(1)]

        rbf_weights = self.compute_rbf_weights(trunk_coords, probe_coords, valid_mask)  # [batch_size, N, probe_count]

        # 5. 残差插值 delta(x) = Σ w_i(x) * r_i
        delta = torch.bmm(rbf_weights, probe_residuals)  # [batch_size, N, 2]

        # 6. 最终预测 y_final = y_base + delta
        y_final = y_base + delta

        # 收集校正信息用于调试和可视化
        correction_info = {
            'y_base': y_base,
            'delta': delta,
            'rbf_weights': rbf_weights,
            'probe_coords': probe_coords,
            'trunk_coords': trunk_coords,
            'y_base_probe': y_base_probe
        }

        return y_final, correction_info


class EnhancedProbeCorrectionModule(ProbeCorrectionModule):
    """
    增强版探针校正模块：支持MLP增强的残差学习
    """

    def __init__(self, cfg: Config):
        super().__init__(cfg)

        # MLP增强网络：从[y_base, trunk_coords, 聚合探针信息] -> delta_enhanced
        input_dim = 2 + 2 + 2  # y_base(2) + trunk_coords(2) + aggregated_probe(2)
        hidden_dim = 128

        self.enhance_mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 2)  # 输出校正项 [dx, dy]
        )

        print(f"   MLP增强: 启用 (输入维度={input_dim}, 隐藏维度={hidden_dim})")

    def aggregate_probe_info(self, rbf_weights: torch.Tensor, probe_residuals: torch.Tensor) -> torch.Tensor:
        """
        聚合探针信息作为全局特征
        Args:
            rbf_weights: [batch_size, N, probe_count] RBF权重
            probe_residuals: [batch_size, probe_count, 2] 探针残差
        Returns:
            aggregated_probe: [batch_size, N, 2] 聚合探针信息
        """
        # 使用注意力权重聚合探针信息
        aggregated_probe = torch.bmm(rbf_weights, probe_residuals)  # [batch_size, N, 2]
        return aggregated_probe

    def forward(self, y_base: torch.Tensor, branch_real: torch.Tensor,
                trunk: torch.Tensor, mask: Optional[torch.Tensor] = None,
                lengths: Optional[torch.Tensor] = None,
                probe_true_values: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        增强版前向传播
        Args:
            probe_true_values: [batch_size, probe_count, 2] 真实探针值（用于计算残差）
        """
        batch_size, N, _ = y_base.shape

        if not self.enable_correction:
            return y_base, {'y_base': y_base, 'delta': torch.zeros_like(y_base)}

        # 1. 提取坐标信息
        probe_coords = self.extract_probe_coordinates(branch_real)  # [batch_size, probe_count, 2]
        trunk_coords = self.extract_trunk_coordinates(trunk)        # [batch_size, N, 2]

        # 2. 在探针位置提取基线预测值
        y_base_probe = []
        for i in range(batch_size):
            if mask is not None:
                probe_mask = mask[i].bool()
                y_base_probe_i = y_base[i][probe_mask]
            else:
                probe_count = probe_coords.shape[1]
                y_base_probe_i = y_base[i, :probe_count]
            y_base_probe.append(y_base_probe_i)

        y_base_probe = torch.stack(y_base_probe, dim=0)  # [batch_size, probe_count, 2]

        # 3. 计算探针残差
        if probe_true_values is not None:
            # 确保所有张量在相同设备上（移动到基线预测设备）
            probe_true_values = probe_true_values.to(y_base_probe.device)
            probe_residuals = probe_true_values - y_base_probe  # [batch_size, probe_count, 2]
        else:
            # 临时方案：假设没有真实探针值
            probe_residuals = torch.zeros_like(y_base_probe)

        # 4. 计算RBF权重
        # 构造有效区域mask，只用于排除padding位置，不使用探针mask
        valid_mask = None
        if lengths is not None:
            max_len = trunk.size(1)
            idx = torch.arange(max_len, device=trunk.device).unsqueeze(0)  # [1, max_len]
            # 确保lengths与idx在相同设备上
            lengths = lengths.to(trunk.device)
            valid_mask = idx < lengths.unsqueeze(1)  # [batch_size, max_len]
            # 截断到实际长度
            valid_mask = valid_mask[:, :trunk.size(1)]

        rbf_weights = self.compute_rbf_weights(trunk_coords, probe_coords, valid_mask)  # [batch_size, N, probe_count]

        # 5. RBF插值得到基础delta
        delta_rbf = torch.bmm(rbf_weights, probe_residuals)  # [batch_size, N, 2]

        # 6. MLP增强：聚合探针信息
        aggregated_probe = self.aggregate_probe_info(rbf_weights, probe_residuals)  # [batch_size, N, 2]

        # 7. 构建MLP输入 [y_base, trunk_coords, aggregated_probe]
        mlp_input = torch.cat([
            y_base,                # [batch_size, N, 2]
            trunk_coords,          # [batch_size, N, 2]
            aggregated_probe       # [batch_size, N, 2]
        ], dim=-1)  # [batch_size, N, 6]

        # 8. MLP预测校正项
        delta_mlp = self.enhance_mlp(mlp_input)  # [batch_size, N, 2]

        # 9. 组合校正项：RBF基础校正 + MLP增强校正
        delta = delta_rbf + delta_mlp

        # 10. 最终预测
        y_final = y_base + delta

        # 收集校正信息
        correction_info = {
            'y_base': y_base,
            'delta_rbf': delta_rbf,
            'delta_mlp': delta_mlp,
            'delta': delta,
            'rbf_weights': rbf_weights,
            'probe_coords': probe_coords,
            'trunk_coords': trunk_coords,
            'probe_residuals': probe_residuals,
            'aggregated_probe': aggregated_probe
        }

        return y_final, correction_info


def create_probe_correction_module(cfg: Config) -> ProbeCorrectionModule:
    """
    工厂函数：根据配置创建探针校正模块
    """
    if cfg.physics.learn_rbf_gamma or cfg.physics.rbf_frequency_aware:
        # 如果使用了高级功能，返回增强版
        return EnhancedProbeCorrectionModule(cfg)
    else:
        # 否则返回基础版
        return ProbeCorrectionModule(cfg)


if __name__ == "__main__":
    # 测试探针校正模块
    cfg = Config()

    # 创建模块
    probe_module = create_probe_correction_module(cfg)

    # 创建测试数据
    batch_size = 2
    probe_count = cfg.deeponet.probe_count  # 50
    N = 100

    y_base = torch.randn(batch_size, N, 2)
    branch_real = torch.randn(batch_size, probe_count * 5)
    trunk = torch.randn(batch_size, N, 3)
    mask = torch.zeros(batch_size, N, dtype=torch.bool)

    # 设置探针mask（假设前probe_count个点是探针）
    probe_count = cfg.deeponet.probe_count  # 50
    for i in range(batch_size):
        mask[i, :probe_count] = True

    print(f"测试数据形状:")
    print(f"  y_base: {y_base.shape}")
    print(f"  branch_real: {branch_real.shape}")
    print(f"  trunk: {trunk.shape}")
    print(f"  mask: {mask.shape}")
    print(f"  探针数量: {probe_count}")
    print(f"  Mask中探针位置数: {mask[0].sum().item()}")

    # 测试前向传播
    try:
        y_final, correction_info = probe_module(y_base, branch_real, trunk, mask)
        print(f"探针校正模块测试成功!")
        print(f"  输出形状: {y_final.shape}")
        print(f"  校正项形状: {correction_info['delta'].shape}")
        print(f"  RBF权重形状: {correction_info['rbf_weights'].shape}")

        # 检查数值范围
        print(f"  输入范围: y_base=[{y_base.min():.3f}, {y_base.max():.3f}]")
        print(f"  校正范围: delta=[{correction_info['delta'].min():.3f}, {correction_info['delta'].max():.3f}]")
        print(f"  输出范围: y_final=[{y_final.min():.3f}, {y_final.max():.3f}]")

    except Exception as e:
        print(f"探针校正模块测试失败: {e}")
        import traceback
        traceback.print_exc()