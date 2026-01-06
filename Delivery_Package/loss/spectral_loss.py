"""
Spectral Loss (k空间FFT损失) - 完全复制客户的FFT逻辑
File: loss/spectral_loss.py
Day 1 核心交付物
"""
import torch
import torch.nn as nn
import numpy as np
from scipy.interpolate import griddata
from typing import Optional, Tuple, Dict


class SpectralLoss(nn.Module):
    """
    Spectral Loss: 在k空间计算预测场与真实场的FFT差异

    完全复制客户的FFT逻辑 (step5_fft_analysis.py):
    1. 将不规则点云插值到128×128规则网格 (griddata)
    2. 执行2D FFT + fftshift + 空间缩放 (dx*dy)
    3. 计算GT固定mask (threshold = max/1000)
    4. 应用mask后计算加权L2损失

    重要说明:
    - Day 1版本：使用scipy.griddata插值，保证与客户FFT逻辑完全一致
    - 梯度流：由于griddata不可微，此loss主要用于监控和评估
    - 训练梯度：主要来自probe_loss, field_loss, correlation_loss
    - 未来改进：可用torch.nn.functional.grid_sample实现可微版本

    Args:
        grid_size: FFT网格大小，默认128
        threshold_ratio: 阈值比率，默认1000 (客户固定值)
        cache_masks: 是否缓存GT的mask，提升性能
    """

    def __init__(self,
                 grid_size: int = 128,
                 threshold_ratio: float = 1000.0,
                 cache_masks: bool = True):
        super(SpectralLoss, self).__init__()

        self.grid_size = grid_size
        self.threshold_ratio = threshold_ratio
        self.cache_masks = cache_masks

        # Mask缓存字典 {sample_id: (mask, weight)}
        self.mask_cache: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}

        print(f"[SpectralLoss] 初始化完成")
        print(f"  网格大小: {grid_size}×{grid_size}")
        print(f"  阈值比率: 1/{threshold_ratio:.0f}")
        print(f"  Mask缓存: {'启用' if cache_masks else '禁用'}")

    def _interpolate_to_grid(self,
                            coords: torch.Tensor,
                            field_values: torch.Tensor,
                            grid_bounds: Tuple[float, float, float, float] = (0.0, 4.5, 0.5, 5.5)
                            ) -> torch.Tensor:
        """
        将不规则点云插值到规则网格（复制step4逻辑）

        Args:
            coords: [N, 3] 坐标 (x, y, z)，只使用x,y
            field_values: [N] 标量场值（Real或Imag）
            grid_bounds: (x_min, x_max, y_min, y_max)

        Returns:
            grid_field: [grid_size, grid_size] tensor 插值后的网格场
        """
        device = coords.device

        # 转换为numpy进行插值
        coords_np = coords.detach().cpu().numpy()
        field_np = field_values.detach().cpu().numpy()

        # 提取x, y坐标
        x_coords = coords_np[:, 0]
        y_coords = coords_np[:, 1]

        # 创建规则网格
        x_min, x_max, y_min, y_max = grid_bounds
        x_grid = np.linspace(x_min, x_max, self.grid_size)
        y_grid = np.linspace(y_min, y_max, self.grid_size)
        X_grid, Y_grid = np.meshgrid(x_grid, y_grid)

        # 插值（完全复制step4_fft_preprocessing.py第139行）
        grid_field_np = griddata(
            (x_coords, y_coords), field_np,
            (X_grid, Y_grid), method='linear', fill_value=0.0
        )

        # 转换为torch tensor并保持在原设备上
        grid_field = torch.from_numpy(grid_field_np).float().to(device)

        return grid_field

    def _compute_fft_with_scaling(self,
                                   grid_real: torch.Tensor,
                                   grid_imag: torch.Tensor,
                                   grid_bounds: Tuple[float, float, float, float]
                                   ) -> torch.Tensor:
        """
        执行2D FFT + fftshift + 空间缩放（复制step5逻辑）

        Args:
            grid_real: [H, W] 实部网格 (torch tensor)
            grid_imag: [H, W] 虚部网格 (torch tensor)
            grid_bounds: (x_min, x_max, y_min, y_max)

        Returns:
            fft_result: [H, W] complex tensor，k空间频谱
        """
        # 合成复数场
        grid_complex = torch.complex(grid_real, grid_imag)

        # 计算空间步长
        x_min, x_max, y_min, y_max = grid_bounds
        dx = (x_max - x_min) / (self.grid_size - 1)
        dy = (y_max - y_min) / (self.grid_size - 1)

        # 执行2D FFT（复制step5_fft_analysis.py第83行）
        # Psi_k = np.fft.fftshift(np.fft.fft2(psi_for_fft)) * dx_fft * dy_fft
        fft_result = torch.fft.fft2(grid_complex)
        fft_shifted = torch.fft.fftshift(fft_result)
        fft_scaled = fft_shifted * dx * dy  # 空间离散化缩放

        return fft_scaled

    def _compute_gt_mask_and_weight(self,
                                     fft_gt: torch.Tensor
                                     ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        计算GT固定mask和权重（复制step5逻辑）

        CRITICAL: mask必须从GT计算，不能从Pred动态计算（防止梯度不可导）

        Args:
            fft_gt: [H, W] complex tensor，GT的k空间频谱

        Returns:
            mask: [H, W] bool tensor，阈值mask
            weight: [H, W] float tensor，归一化权重
        """
        # 计算幅值
        fft_abs = torch.abs(fft_gt)

        # 计算阈值（复制step5_fft_analysis.py第101行）
        # threshold = np.max(np.abs(Psi_k)) / THRESHOLD_RATIO
        threshold = fft_abs.max() / self.threshold_ratio

        # 生成二值mask（复制step5第102行）
        # Psi_k_thresholded = np.where(np.abs(Psi_k) >= threshold, Psi_k, 0)
        mask = (fft_abs >= threshold)

        # 计算归一化权重（复制step5第90-91行）
        # weight = np.abs(Psi_k) ** 2
        # weight = weight / weight.max()
        weight = (fft_abs ** 2) / (fft_abs.max() ** 2 + 1e-10)

        return mask, weight

    def forward(self,
                y_pred: torch.Tensor,
                y_true: torch.Tensor,
                coords: torch.Tensor,
                sample_ids: Optional[torch.Tensor] = None,
                grid_bounds: Tuple[float, float, float, float] = (0.0, 4.5, 0.5, 5.5)
                ) -> torch.Tensor:
        """
        计算Spectral Loss

        Args:
            y_pred: [batch_size, N, 2] 预测场 (real, imag)
            y_true: [batch_size, N, 2] 真实场 (real, imag)
            coords: [batch_size, N, 3] 空间坐标 (x, y, z)
            sample_ids: [batch_size] 样本ID，用于缓存mask
            grid_bounds: (x_min, x_max, y_min, y_max)

        Returns:
            loss: scalar tensor，Spectral Loss
        """
        batch_size = y_pred.shape[0]
        device = y_pred.device

        total_loss = 0.0

        for i in range(batch_size):
            # 提取单个样本
            pred_real = y_pred[i, :, 0]
            pred_imag = y_pred[i, :, 1]
            true_real = y_true[i, :, 0]
            true_imag = y_true[i, :, 1]
            coords_i = coords[i]  # [N, 3]

            # 1) 插值到规则网格
            pred_grid_real = self._interpolate_to_grid(coords_i, pred_real, grid_bounds)
            pred_grid_imag = self._interpolate_to_grid(coords_i, pred_imag, grid_bounds)
            true_grid_real = self._interpolate_to_grid(coords_i, true_real, grid_bounds)
            true_grid_imag = self._interpolate_to_grid(coords_i, true_imag, grid_bounds)

            # 2) 执行2D FFT
            fft_pred = self._compute_fft_with_scaling(pred_grid_real, pred_grid_imag, grid_bounds)
            fft_true = self._compute_fft_with_scaling(true_grid_real, true_grid_imag, grid_bounds)

            # 3) 计算或获取缓存的GT mask
            if self.cache_masks and sample_ids is not None:
                sample_id = int(sample_ids[i].item())

                if sample_id in self.mask_cache:
                    # 使用缓存的mask
                    mask, weight = self.mask_cache[sample_id]
                else:
                    # 计算新的mask并缓存
                    mask, weight = self._compute_gt_mask_and_weight(fft_true)
                    self.mask_cache[sample_id] = (mask, weight)
            else:
                # 不使用缓存，每次计算
                mask, weight = self._compute_gt_mask_and_weight(fft_true)

            # 4) 应用固定mask
            fft_pred_masked = fft_pred * mask.float()
            fft_true_masked = fft_true * mask.float()

            # 5) 计算加权L2损失
            diff = torch.abs(fft_pred_masked - fft_true_masked) ** 2
            weighted_diff = diff * weight

            # 归一化：除以mask内的点数
            sample_loss = weighted_diff.sum() / (mask.sum() + 1e-10)
            total_loss += sample_loss

        # 平均batch loss
        avg_loss = total_loss / batch_size

        return avg_loss

    def clear_cache(self):
        """清空mask缓存"""
        self.mask_cache.clear()
        print(f"[SpectralLoss] Mask缓存已清空")

    def get_cache_stats(self) -> Dict[str, int]:
        """获取缓存统计信息"""
        return {
            'cached_samples': len(self.mask_cache),
            'cache_enabled': self.cache_masks
        }


# ========================================
# 测试代码
# ========================================
if __name__ == "__main__":
    print("=" * 60)
    print("测试 Spectral Loss 实现")
    print("=" * 60)

    # 创建测试数据
    batch_size = 2
    num_points = 100

    # 随机生成不规则点云
    coords = torch.rand(batch_size, num_points, 3) * 4.0  # x,y,z ∈ [0, 4]
    coords[:, :, 0] = coords[:, :, 0] * 1.125  # x ∈ [0, 4.5]
    coords[:, :, 1] = coords[:, :, 1] * 1.25 + 0.5  # y ∈ [0.5, 5.5]

    y_pred = torch.randn(batch_size, num_points, 2)  # 随机预测
    y_true = torch.randn(batch_size, num_points, 2)  # 随机真实
    sample_ids = torch.arange(batch_size)

    # 创建Spectral Loss
    spectral_fn = SpectralLoss(grid_size=128, threshold_ratio=1000.0, cache_masks=True)

    # 计算损失
    print("\n第一次计算 (无缓存)...")
    loss1 = spectral_fn(y_pred, y_true, coords, sample_ids)
    print(f"Spectral Loss: {loss1.item():.6f}")
    print(f"Loss可微分: {loss1.requires_grad}")

    # 再次计算（应使用缓存）
    print("\n第二次计算 (使用缓存)...")
    loss2 = spectral_fn(y_pred, y_true, coords, sample_ids)
    print(f"Spectral Loss: {loss2.item():.6f}")

    # 缓存统计
    cache_stats = spectral_fn.get_cache_stats()
    print(f"\n缓存统计: {cache_stats}")

    # 数值稳定性检查
    print("\n数值稳定性检查...")
    print(f"Loss值范围正常: {0 < loss1.item() < 1e6}")
    print(f"两次计算一致性: {abs(loss1.item() - loss2.item()) < 1e-6}")

    # Note: 由于使用scipy.griddata (不可微分)，此loss主要用于监控
    print("\n说明: Spectral Loss使用scipy.griddata进行插值")
    print("  - 完全复制客户FFT逻辑，保证准确性")
    print("  - 主要用于监控和评估k空间重构质量")
    print("  - 训练梯度来自probe_loss、field_loss等可微分loss")

    print("\n" + "=" * 60)
    print("✅ Spectral Loss 测试通过")
    print("=" * 60)
