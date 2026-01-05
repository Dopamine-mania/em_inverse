"""
改进版DeepONet训练脚本 - 修复版本
1. 真正的批处理训练（支持变长序列）
2. 平衡的损失权重
3. 改进的学习率调度
4. 优化的数据处理
5. 自定义collate函数处理变长序列
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import argparse
import os
import multiprocessing as mp
from pathlib import Path
import sys
import time
import json
from typing import Tuple, Dict, List, Optional
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter  # 添加TensorBoard支持

# 设置CUDA环境变量以优化多GPU性能
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,3'  # 排除被占用的GPU 2
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

# 修复PyTorch 2.0+的CUDA generator问题
# 设置默认随机数生成器为CPU
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
    torch.cuda.empty_cache()  # 清理GPU缓存

# 设置multiprocessing start method为spawn以支持CUDA
if __name__ == '__main__':
    try:
        mp.set_start_method('spawn', force=True)
        print("Multiprocessing start method set to 'spawn' for CUDA compatibility")
    except RuntimeError:
        print("Multiprocessing start method already set")

# 强制设置PyTorch的全局generator为CPU设备
import torch.utils.data as data
data.get_worker_info = lambda: None  # 禁用worker info

# 添加模块路径
sys.path.append(str(Path(__file__).parent))

from config import Config
from data.dataset import MaskedDeepONetDataset
from model.model import create_pytorch_dual_branch_deeponet  # 原始版本，保持兼容性
from model.enhanced_deeponet import create_enhanced_deeponet  # 增强版本（推荐使用）
from loss.loss import PyTorchLossFunction
from utils.utils import (
    save_checkpoint, load_checkpoint, visualize_predictions,
    calculate_metrics, print_metrics, set_random_seed, get_model_summary
)


class DynamicDeepONetDataset(Dataset):
    """动态加载的DeepONet数据集类 - 避免一次性处理所有数据"""

    def __init__(self, samples_list: List[Dict], cfg: Config):
        self.samples_list = samples_list
        self.cfg = cfg
        self.dataset = MaskedDeepONetDataset(cfg)

    def __len__(self):
        return len(self.samples_list)

    def __getitem__(self, idx):
        sample = self.samples_list[idx]
        try:
            # 动态处理单个样本 - 添加固定探针选择选项用于调试
            freq_idx = sample.get('freq_idx', 0)

            # 调试模式：使用固定的sample_idx确保探针选择一致
            # 正常训练：使用随机sample_idx增加数据多样性
            import random
            sample_idx = idx if getattr(self, '_fixed_probe_mode', False) else random.randint(0, 1000000)

            if getattr(self, '_fixed_probe_mode', False):
                print(f"[排查] 固定探针模式: sample_idx={sample_idx} (idx={idx})")

            # 修改：dataset.prepare_single_sample现在返回单branch格式
            branch_input, trunk_input, target_output, mask, probe_coords_3d = self.dataset.prepare_single_sample(
                sample['file'], freq_idx=freq_idx, sample_idx=sample_idx
            )

            # 转换为张量并重塑 (单Branch格式: 25*4+1=101维)
            expected_branch_size = self.cfg.deeponet.probe_count * 4 + 1  # 修复：单Branch输入 (x,y,real,imag) * 25 + freq
            branch_tensor = torch.from_numpy(branch_input).float().view(-1)  # [101]
            trunk_tensor = torch.from_numpy(trunk_input).float()  # [N, 3]
            target_tensor = torch.from_numpy(target_output).float()  # [N, 2]
            mask_tensor = torch.from_numpy(mask).bool()  # [N]
            probe_coords_tensor = torch.from_numpy(probe_coords_3d).float()  # [25, 3]

            return {
                'branch_input': branch_tensor,  # [101] 单Branch输入
                'trunk': trunk_tensor,  # [N, 3] = (x, y, frequency)
                'y': target_tensor,     # [N, 2] = (real, imag)
                'mask': mask_tensor,    # [N] = probe positions in trunk
                'probe_coords': probe_coords_tensor,  # [25, 3] 探针坐标（用于Spectral Loss）
                'sample_idx': idx  # 样本索引（用于Spectral Loss缓存）
            }
        except Exception as e:
            print(f"动态加载样本失败 {sample.get('source_file', sample['file'])}: {e}")
            # 返回一个空的样本而不是抛出异常
            expected_branch_size = self.cfg.deeponet.probe_count * 4 + 1  # 单Branch格式
            print(f"[排查] 返回空样本: branch_size={expected_branch_size}")
            return {
                'branch_input': torch.zeros(expected_branch_size),
                'trunk': torch.zeros(1, 3),
                'y': torch.zeros(1, 2),
                'mask': torch.zeros(1, dtype=torch.bool),
                'probe_coords': torch.zeros(self.cfg.data.num_probes, 3),
                'sample_idx': idx
            }


def collate_fn(batch, cfg=None):
    """
    自定义collate函数处理变长序列 - 单Branch版本，支持Spectral Loss
    Args:
        batch: List of Dict with keys: 'branch_input', 'trunk', 'y', 'mask', 'probe_coords', 'sample_idx'
        cfg: Config 配置对象，用于控制警告行为
    Returns:
        Dict with padded sequences, probe coords, and sample IDs for Spectral Loss
    """
    batch_size = len(batch)

    # 分离各个字段
    branch_data = [item['branch_input'] for item in batch]  # List of [101]
    trunk_data = [item['trunk'] for item in batch]          # List of [N_i, 3]
    y_data = [item['y'] for item in batch]                 # List of [N_i, 2]
    masks = [item['mask'] for item in batch]               # List of [N_i]
    probe_coords_data = [item['probe_coords'] for item in batch]  # List of [25, 3]
    sample_ids = [item['sample_idx'] for item in batch]   # List of int

    # Branch数据堆叠（都是固定长度）
    branch_tensor = torch.stack(branch_data, dim=0)  # [batch_size, 101]

    # 找到trunk和y的最大长度
    max_trunk_length = max(item.shape[0] for item in trunk_data)
    max_mask_length = max(item.shape[0] for item in masks)

    # 初始化填充后的张量
    trunk_padded = torch.zeros(batch_size, max_trunk_length, 3)  # [batch, max_trunk_len, 3]
    y_padded = torch.zeros(batch_size, max_trunk_length, 2)       # [batch, max_trunk_len, 2]
    mask_padded = torch.zeros(batch_size, max_mask_length, dtype=torch.bool)  # [batch, max_mask_len]

    # 提取真实探针值 - 用于RBF校正
    probe_true_values_list = []
    expected_probe_count = None  # 期望的探针数量
    enable_warnings = cfg is not None and cfg.physics.probe_alignment_warning

    # 填充数据
    trunk_lengths = []
    for i, (trunk, y, mask) in enumerate(zip(trunk_data, y_data, masks)):
        trunk_length = trunk.shape[0]
        mask_length = mask.shape[0]
        trunk_lengths.append(trunk_length)

        # 填充trunk和y数据
        trunk_padded[i, :trunk_length] = trunk
        y_padded[i, :trunk_length] = y

        # 填充mask数据 - 使用mask的实际长度
        mask_padded[i, :mask_length] = mask

        # 提取探针位置的真实值
        probe_mask = mask.bool()  # [total_points_i]
        probe_values_i = y[probe_mask]  # [num_probes, 2]

        # 检查探针对齐情况
        actual_probe_count = probe_values_i.shape[0]

        if expected_probe_count is None:
            # 第一个样本设置期望数量
            expected_probe_count = actual_probe_count

        if actual_probe_count == 0:
            if enable_warnings:
                print(f"警告: 样本 {i} 中没有探针点 (mask为空)，将使用零填充")
            # 使用配置的探针数量作为期望数量
            if expected_probe_count == 0:
                expected_probe_count = 50  # 从配置推断，或硬编码默认值
                if cfg is not None:
                    expected_probe_count = cfg.deeponet.probe_count
        elif actual_probe_count != expected_probe_count:
            if enable_warnings:
                print(f"警告: 样本 {i} 探针数量不匹配 - 期望: {expected_probe_count}, 实际: {actual_probe_count}")

            if actual_probe_count > expected_probe_count:
                if enable_warnings:
                    print(f"   截断前 {actual_probe_count - expected_probe_count} 个探针")
                probe_values_i = probe_values_i[:expected_probe_count]
            else:
                if enable_warnings:
                    print(f"   填充 {expected_probe_count - actual_probe_count} 个零值探针")
                padding = torch.zeros(expected_probe_count - actual_probe_count, 2)
                probe_values_i = torch.cat([probe_values_i, padding], dim=0)

        probe_true_values_list.append(probe_values_i)

    # 最终检查探针数量一致性
    final_probe_counts = [values.shape[0] for values in probe_true_values_list]
    if not all(count == final_probe_counts[0] for count in final_probe_counts):
        if enable_warnings:
            print(f"错误: 探针数量不一致 - 各样本探针数: {final_probe_counts}")
            print(f"   这可能导致RBF校正错位，请检查数据生成过程")

        # 强制对齐到最小数量，避免更严重的错位
        min_probe_count = min(final_probe_counts)
        if enable_warnings:
            print(f"   强制对齐到最小探针数量: {min_probe_count}")
        probe_true_values_list = [values[:min_probe_count] for values in probe_true_values_list]

    # 堆叠探针真实值
    probe_true_values = torch.stack(probe_true_values_list, dim=0)  # [batch_size, probe_count, 2]

    # 堆叠探针坐标和样本ID（用于Spectral Loss）
    probe_coords_tensor = torch.stack(probe_coords_data, dim=0)  # [batch_size, 25, 3]
    sample_ids_tensor = torch.tensor(sample_ids, dtype=torch.long)  # [batch_size]

    # 最终验证（仅在启用警告时显示）
    if enable_warnings:
        batch_size, actual_probe_count, _ = probe_true_values.shape
        print(f"探针数据验证:")
        print(f"   批次大小: {batch_size}")
        print(f"   探针数量: {actual_probe_count}")
        if probe_true_values.numel() > 0:
            print(f"   探针真值范围: [{probe_true_values.min():.6f}, {probe_true_values.max():.6f}]")
        else:
            print(f"   探针真值范围: 空张量")

    return {
        'branch_input': branch_tensor,  # [batch_size, 101] 单Branch输入
        'trunk': trunk_padded,
        'y': y_padded,
        'mask': mask_padded,
        'lengths': torch.tensor(trunk_lengths),
        'probe_true_values': probe_true_values,  # 真实探针值用于RBF校正
        'probe_coords': probe_coords_tensor,  # [batch_size, 25, 3] 用于Spectral Loss
        'sample_ids': sample_ids_tensor  # [batch_size] 用于Spectral Loss缓存
    }


class DeepONetBatchDataset(Dataset):
    """真正的批处理数据集"""

    def __init__(self, branch_data, trunk_data, y_data, masks):
        self.branch_data = branch_data  # List of [1, 50, 5]
        self.trunk_data = trunk_data    # List of [N_i, 3]
        self.y_data = y_data           # List of [N_i, 2]
        self.masks = masks             # List of [N_i]

    def __len__(self):
        return len(self.branch_data)

    def __getitem__(self, idx):
        branch_data = self.branch_data[idx]  # [1, 50, 5] 或 [50, 5] 或其他形状

        # 确保branch_data是正确的维度
        if branch_data.dim() == 3:
            # [1, 50, 5] -> [50, 5]
            branch_data = branch_data.squeeze(0)

        # 重塑为[250]
        branch_data = branch_data.view(-1)  # [50, 5] -> [250]

        return {
            'branch': branch_data,              # [250]
            'trunk': self.trunk_data[idx],      # [N_i, 3]
            'y': self.y_data[idx],             # [N_i, 2]
            'mask': self.masks[idx]            # [N_i]
        }


# 在类开始处添加配置参数
class ImprovedDeepONetTrainer:
    """改进版DeepONet训练器 - 修复版本"""

    def __init__(self, cfg: Config):
        self.cfg = cfg

        # 性能优化配置
        self.max_sample_length = 1000  # 可配置的最大样本长度
        self.use_adaptive_sampling = True  # 是否使用自适应采样
        self.cfg = cfg
        # 检查可用的GPU设备
        print(f"=== 检查GPU可用性 ===")
        print(f"CUDA可用: {torch.cuda.is_available()}")
        print(f"GPU数量: {torch.cuda.device_count()}")

        available_gpus = []
        for i in range(torch.cuda.device_count()):  # 检查所有可用GPU
            try:
                # 测试GPU是否可用
                torch.cuda.set_device(i)
                test_tensor = torch.cuda.FloatTensor(1)
                available_gpus.append(i)
                print(f"GPU {i}: {torch.cuda.get_device_name(i)} - 可用")
            except Exception as e:
                print(f"GPU {i}: 不可用 - {e}")

        if not available_gpus:
            print("未找到可用的GPU，使用CPU")
            self.device = torch.device('cpu')
            self.multi_gpu = False
        else:
            # 使用第一个可用的GPU作为主设备
            main_gpu = available_gpus[0]
            self.device = torch.device(f'cuda:{main_gpu}')
            print(f"使用主GPU: {main_gpu}")
            print(f"可用GPU列表: {available_gpus}")

            # 如果有多个可用GPU，启用DataParallel
            if len(available_gpus) > 1:
                print(f"启用多GPU训练 (使用 {len(available_gpus)} 个GPU)")
                self.multi_gpu = True

                # 由于设置了CUDA_VISIBLE_DEVICES='0,1,3'，PyTorch看到的GPU索引是0,1,2
                # 但我们需要跳过原来被占用的GPU 2，所以使用所有可用GPU
                self.gpu_ids = list(range(len(available_gpus)))  # 使用0,1,2对应实际的0,1,3
                print(f"使用GPU索引映射: {self.gpu_ids} (对应实际GPU: [0,1,3])")
            else:
                print("使用单GPU训练")
                self.multi_gpu = False

        # 设置随机种子
        set_random_seed(42)

        # 初始化模型
        # 使用增强版DeepONet，支持残差连接、注意力、Dropout等高级功能
        print(f"使用增强版DeepONet网络")
        print(f"网络预设: {cfg.deeponet.network_preset}")
        print(f"激活函数: {cfg.deeponet.activation}")
        print(f"Dropout率: {cfg.deeponet.dropout_rate}")
        print(f"残差连接: {'启用' if cfg.deeponet.use_residual else '禁用'}")
        print(f"注意力机制: {'启用' if cfg.deeponet.use_attention else '禁用'}")

        self.model = create_enhanced_deeponet(cfg)

        # 增强的模型架构摘要
        print("\n=== 模型架构摘要 ===")
        print(f"网络预设: {cfg.deeponet.network_preset}")
        print(f"隐藏层结构: {cfg.deeponet.hidden_layers}")
        print(f"输出维度: {cfg.deeponet.output_dim}")
        print(f"激活函数: {cfg.deeponet.activation}")
        print(f"Dropout率: {cfg.deeponet.dropout_rate}")
        print(f"注意力机制: {'启用' if cfg.deeponet.use_attention else '禁用'}")
        print(f"残差连接: {'启用' if cfg.deeponet.use_residual else '禁用'}")
        print(f"RBF校正: {'启用' if cfg.physics.enable_rbf_correction else '禁用'}")
        print(f"探针数量: {cfg.data.num_probes}")

        # 显示网络配置信息
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"总参数数量: {total_params:,}")
        print(f"可训练参数: {trainable_params:,}")
        print(f"网络结构: {cfg.deeponet.hidden_layers} -> {cfg.deeponet.output_dim}")
        print("=" * 40)

        self.model.to(self.device)

        # 启用多GPU训练
        if hasattr(self, 'multi_gpu') and self.multi_gpu:
            print(f"正在将模型分布到GPU: {self.gpu_ids}")
            self.model = torch.nn.DataParallel(self.model, device_ids=self.gpu_ids)
            print(f"DataParallel包装完成")
            print(f"模型类型: {type(self.model)}")
            print(f"模型参数总数: {sum(p.numel() for p in self.model.parameters()):,}")
            print(f"可训练参数: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")
        else:
            print("使用单GPU模式")
            print(f"模型参数总数: {sum(p.numel() for p in self.model.parameters()):,}")
            print(f"可训练参数: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")

        # 优化的优化器和调度器（移除verbose参数）
        if hasattr(self, 'multi_gpu') and self.multi_gpu:
            # DataParallel会包装模型，需要访问.module属性
            self.optimizer = optim.Adam(
                self.model.module.parameters(),
                lr=cfg.training.learning_rate,
                weight_decay=1e-5  # 恢复正常权重衰减
            )
        else:
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=cfg.training.learning_rate,
                weight_decay=1e-5  # 恢复正常权重衰减
            )

        # 学习率调度器 - 修复版本兼容性问题
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.8, patience=20
        )

        # 损失函数
        self.loss_fn = PyTorchLossFunction(cfg)

        # 初始化TensorBoard writer
        log_dir = Path(self.cfg.visualization.output_dir) / "tensorboard"
        log_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir=log_dir)
        print(f"TensorBoard日志目录: {log_dir}")
        print(f"启动命令: tensorboard --logdir {log_dir}")

        print(f"改进版训练器初始化完成")
        print(f"使用设备: {self.device}")
        print(f"损失权重 - Probe: {cfg.physics.probe_loss_weight}, "
              f"Field: {cfg.physics.field_loss_weight}, "
              f"Smooth: {cfg.physics.smoothness_loss_weight}")
        get_model_summary(self.model)

    def train_epoch_optimized(self, dataloader, epoch: int) -> Tuple[float, Dict[str, float]]:
        """优化版训练epoch - 真正的批量处理"""
        self.model.train()

        # 移除每个epoch的torch.manual_seed(42)调用，避免随机性一致导致dropout失效
        # 全局随机种子已在训练器初始化时通过set_random_seed(42)设置一次

        epoch_losses = {'total': 0, 'probe': 0, 'field': 0, 'correlation': 0, 'smooth': 0, 'spectral_loss': 0}
        num_batches = 0

        print(f"[DEBUG] Starting epoch {epoch}, dataloader length: {len(dataloader)}")

        try:
            # 计时变量
            data_load_time = 0
            model_forward_time = 0
            loss_compute_time = 0
            backward_time = 0
            optimizer_time = 0

            for batch_idx, batch in enumerate(dataloader):
                # 数据移动计时开始
                start_time = time.time()

                # 移动数据到设备 - 单Branch格式
                branch_input_batch = batch['branch_input'].to(self.device)  # [batch_size, 101]
                trunk_batch = batch['trunk'].to(self.device)             # [batch_size, max_len, 3]
                y_batch = batch['y'].to(self.device)                    # [batch_size, max_len, 2]
                mask_batch = batch['mask'].to(self.device)               # [batch_size, max_len]
                lengths = batch['lengths']                               # [batch_size]
                probe_coords = batch['probe_coords'].to(self.device)     # [batch_size, 25, 3]
                sample_ids = batch['sample_ids'].to(self.device)         # [batch_size]

                data_load_time += time.time() - start_time

                current_batch_size = branch_input_batch.shape[0]

                # 清零梯度
                self.optimizer.zero_grad()

                # 优化版批量处理：移除样本级别的for循环
                batch_loss = 0.0
                loss_dict_batch = {'probe_loss': 0, 'field_loss': 0, 'smooth_loss': 0}

                # 模型前向传播计时开始
                forward_start = time.time()

                # 直接批量预测 - 单Branch输入，传递真实探针值用于RBF校正
                probe_true_values = batch.get('probe_true_values', None)  # [batch_size, probe_count, 2]

                if probe_true_values is not None:
                    y_pred = self.model(branch_input_batch, trunk_batch,
                                      mask=mask_batch, lengths=lengths,
                                      probe_true_values=probe_true_values)
                else:
                    # 向后兼容：如果没有探针值，使用原来的调用方式
                    y_pred = self.model(branch_input_batch, trunk_batch,
                                      mask=mask_batch, lengths=lengths)

                model_forward_time += time.time() - forward_start

                # 损失计算计时开始
                loss_start = time.time()

                # 批量损失计算 - 传递coords和sample_ids用于Spectral Loss
                loss, loss_dict = self.loss_fn(
                    y_pred, y_batch,
                    [mask_batch[i] for i in range(mask_batch.shape[0])],
                    lengths,
                    coords=trunk_batch,      # [batch_size, max_len, 3] 空间坐标
                    sample_ids=sample_ids    # [batch_size] 样本ID用于Spectral Loss缓存
                )

                # 探针mask统计 - 只在第一个批次打印
                if batch_idx == 0:
                    for i in range(min(2, mask_batch.shape[0])):  # 只检查前2个样本
                        mask_sum = mask_batch[i].sum().item()
                        expected_probes = self.cfg.data.num_probes
                        match_status = "OK" if mask_sum == expected_probes else "FAIL"

                loss_compute_time += time.time() - loss_start

                # 反向传播计时开始
                backward_start = time.time()

                # 反向传播
                loss.backward()

                # 梯度裁剪 - 防止梯度爆炸，处理DataParallel情况
                if hasattr(self, 'multi_gpu') and self.multi_gpu:
                    torch.nn.utils.clip_grad_norm_(self.model.module.parameters(), max_norm=1.0)
                else:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                backward_time += time.time() - backward_start

                # 优化器计时开始
                optimizer_start = time.time()

                # 更新参数
                self.optimizer.step()

                optimizer_time += time.time() - optimizer_start

                # 记录损失
                batch_loss = loss.item()
                epoch_losses['total'] += batch_loss

                # 安全地提取损失值（处理tensor和float类型）
                if hasattr(loss_dict['probe_loss'], 'item'):
                    epoch_losses['probe'] += loss_dict['probe_loss'].item()
                else:
                    epoch_losses['probe'] += float(loss_dict['probe_loss'])

                if hasattr(loss_dict['field_loss'], 'item'):
                    epoch_losses['field'] += loss_dict['field_loss'].item()
                else:
                    epoch_losses['field'] += float(loss_dict['field_loss'])

                if hasattr(loss_dict['correlation_loss'], 'item'):
                    epoch_losses['correlation'] += loss_dict['correlation_loss'].item()
                else:
                    epoch_losses['correlation'] += float(loss_dict['correlation_loss'])

                if hasattr(loss_dict['smooth_loss'], 'item'):
                    epoch_losses['smooth'] += loss_dict['smooth_loss'].item()
                else:
                    epoch_losses['smooth'] += float(loss_dict['smooth_loss'])

                # 🔧 关键修复：添加spectral loss聚合
                if hasattr(loss_dict['spectral_loss'], 'item'):
                    epoch_losses['spectral_loss'] += loss_dict['spectral_loss'].item()
                else:
                    epoch_losses['spectral_loss'] += float(loss_dict['spectral_loss'])

                num_batches += 1

                # 打印进度和计时信息
                if batch_idx % max(1, len(dataloader) // 10) == 0:
                    # 获取原始损失值
                    probe_val = loss_dict['probe_loss'].item() if hasattr(loss_dict['probe_loss'], 'item') else float(loss_dict['probe_loss'])
                    field_val = loss_dict['field_loss'].item() if hasattr(loss_dict['field_loss'], 'item') else float(loss_dict['field_loss'])
                    correlation_val = loss_dict['correlation_loss'].item() if hasattr(loss_dict['correlation_loss'], 'item') else float(loss_dict['correlation_loss'])
                    smooth_val = loss_dict['smooth_loss'].item() if hasattr(loss_dict['smooth_loss'], 'item') else float(loss_dict['smooth_loss'])

                    # 计算加权贡献值（实际贡献到总loss的值）
                    probe_contribution = self.cfg.physics.probe_loss_weight * probe_val
                    field_contribution = self.cfg.physics.field_loss_weight * field_val
                    correlation_contribution = self.cfg.physics.correlation_loss_weight * correlation_val
                    smooth_contribution = self.cfg.physics.smoothness_loss_weight * smooth_val

                    # 输出加权后的实际贡献值 - 增强探针监控
                    print(f"Epoch {epoch + 1}/{self.cfg.training.epochs} "
                          f"Batch{batch_idx}/{len(dataloader)} "
                          f"探针原始MSE: {probe_val:.6f} → 加权: {probe_contribution:.4f}, "
                          f"场原始MSE: {field_val:.6f} → 加权: {field_contribution:.4f}, "
                          f"相关性: {correlation_contribution:.4f}, "
                          f"总loss: {batch_loss:.4f}")

            # 平均epoch损失
            if num_batches > 0:
                for key in epoch_losses:
                    epoch_losses[key] /= num_batches
            else:
                print(f"[WARNING] No batches processed in epoch {epoch}")
                return 0.0, epoch_losses

            return epoch_losses['total'], epoch_losses

        except Exception as e:
            print(f"[DEBUG] Exception during optimized training: {e}")
            print(f"[DEBUG] Exception type: {type(e)}")
            raise e

    def save_training_visualizations(self, epoch, train_loader, test_loader, train_loss, test_loss):
        """每2个epoch保存训练可视化图片 - 改进版 (2x4布局)"""
        self.model.eval()

        with torch.no_grad():
            # 随机选择不同的批次来显示不同构型
            import random
            train_random_idx = random.randint(0, len(train_loader) - 1)
            test_random_idx = random.randint(0, len(test_loader) - 1)

            # 获取随机批次
            train_batches = list(train_loader)
            test_batches = list(test_loader)
            train_batch = train_batches[train_random_idx]
            test_batch = test_batches[test_random_idx]

            # 处理训练数据 - 随机选择一个样本（单Branch格式）
            sample_idx = random.randint(0, train_batch['branch_input'].shape[0] - 1)
            train_branch_input = train_batch['branch_input'][sample_idx:sample_idx+1].to(self.device)
            train_trunk = train_batch['trunk'][sample_idx:sample_idx+1].to(self.device)
            train_y = train_batch['y'][sample_idx:sample_idx+1].to(self.device)
            train_mask = train_batch['mask'][sample_idx:sample_idx+1].to(self.device)
            train_lengths = train_batch['lengths'][sample_idx:sample_idx+1]

            # 处理测试数据 - 随机选择一个样本（单Branch格式）
            test_sample_idx = random.randint(0, test_batch['branch_input'].shape[0] - 1)
            test_branch_input = test_batch['branch_input'][test_sample_idx:test_sample_idx+1].to(self.device)
            test_trunk = test_batch['trunk'][test_sample_idx:test_sample_idx+1].to(self.device)
            test_y = test_batch['y'][test_sample_idx:test_sample_idx+1].to(self.device)
            test_mask = test_batch['mask'][test_sample_idx:test_sample_idx+1].to(self.device)
            test_lengths = test_batch['lengths'][test_sample_idx:test_sample_idx+1]

            # 创建可视化图片 - 2x4布局
            import matplotlib.pyplot as plt
            import matplotlib
            matplotlib.use('Agg')  # 使用非交互式后端
            from matplotlib.patches import Circle
            import matplotlib.patches as patches

            # 字体设置已通过utils模块全局设置，这里确保一下
            if 'SimHei' not in plt.rcParams['font.sans-serif']:
                plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
            if plt.rcParams['axes.unicode_minus']:
                plt.rcParams['axes.unicode_minus'] = False

            fig, axes = plt.subplots(2, 4, figsize=(24, 12))
            fig.suptitle(f'Epoch {epoch} - Training Progress (构型: Train样本{sample_idx+1}, Test样本{test_sample_idx+1})\n'
                        f'Train Loss: {train_loss:.6f}, Test Loss: {test_loss:.6f}', fontsize=14)

            # 训练集可视化
            train_length = train_lengths[0].item()
            train_coords = train_trunk[0, :train_length, :2].cpu().numpy()
            train_true_real = train_y[0, :train_length, 0].cpu().numpy()  # 实部
            train_true_imag = train_y[0, :train_length, 1].cpu().numpy()  # 虚部

            # 使用正确的长度进行预测（单Branch格式）
            train_trunk_single = train_trunk[0, :train_length].unsqueeze(0)
            train_pred_single = self._predict_single_sample(train_branch_input, train_trunk_single)
            train_pred_real = train_pred_single[:, 0]  # 预测实部
            train_pred_imag = train_pred_single[:, 1]  # 预测虚部

            train_error_real = np.abs(train_true_real - train_pred_real)
            train_error_imag = np.abs(train_true_imag - train_pred_imag)

            # 从branch数据中提取探针位置（单Branch格式：[x1,y1,r1,i1,...,freq]）
            train_branch_data = train_branch_input[0].cpu().numpy()
            # 计算实际的探针数量：(branch数据长度-1)/4，最后1位是频率
            actual_num_probes = (len(train_branch_data) - 1) // 4
            # 提取x,y坐标：每4个元素取前2个
            train_probe_coords = []
            for i in range(actual_num_probes):
                train_probe_coords.append([train_branch_data[i*4], train_branch_data[i*4+1]])
            train_probe_coords = np.array(train_probe_coords)  # [num_probes, 2]

            # 训练集 - 真实值实部 (带colorbar)
            scatter1 = axes[0, 0].scatter(train_coords[:, 0], train_coords[:, 1],
                                        c=train_true_real, cmap='viridis', s=5, alpha=0.6)
            axes[0, 0].set_title('Training Set - True Real Values')
            axes[0, 0].set_xlabel('X')
            axes[0, 0].set_ylabel('Y')
            plt.colorbar(scatter1, ax=axes[0, 0], label='真实场强(实部)')

            # 添加探针位置标注
            self._add_probe_markers(axes[0, 0], train_probe_coords)  # 默认显示所有探针

            # 添加数据点的凸边界
            self._add_convex_boundary(axes[0, 0], train_coords, alpha=0.3, color='blue', label='数据边界')

            # 添加探针的凸边界
            self._add_convex_boundary(axes[0, 0], train_probe_coords, alpha=0.5, color='green', label='探针边界')

            # 添加图例
            axes[0, 0].legend(loc='upper right', fontsize=8)

            # 训练集 - 预测值实部 (带colorbar)
            scatter2 = axes[0, 1].scatter(train_coords[:, 0], train_coords[:, 1],
                                        c=train_pred_real, cmap='viridis', s=5, alpha=0.6)
            axes[0, 1].set_title('Training Set - Predicted Real Values')
            axes[0, 1].set_xlabel('X')
            axes[0, 1].set_ylabel('Y')
            plt.colorbar(scatter2, ax=axes[0, 1], label='预测场强(实部)')

            # 添加探针位置和边界
            self._add_probe_markers(axes[0, 1], train_probe_coords)  # 默认显示所有探针
            self._add_convex_boundary(axes[0, 1], train_coords, alpha=0.3, color='blue', label='数据边界')
            self._add_convex_boundary(axes[0, 1], train_probe_coords, alpha=0.5, color='green', label='探针边界')

            # 训练集 - 误差实部 (带colorbar)
            scatter3 = axes[0, 2].scatter(train_coords[:, 0], train_coords[:, 1],
                                        c=train_error_real, cmap='Reds', s=5, alpha=0.6)
            axes[0, 2].set_title(f'Training Set - Real Error (MAE: {np.mean(train_error_real):.6f})')
            axes[0, 2].set_xlabel('X')
            axes[0, 2].set_ylabel('Y')
            plt.colorbar(scatter3, ax=axes[0, 2], label='实部误差')

            # 训练集 - 实部相关性散点图
            correlation_real = np.corrcoef(train_true_real, train_pred_real)[0, 1]
            axes[0, 3].scatter(train_true_real, train_pred_real, alpha=0.6, s=2)
            axes[0, 3].plot([train_true_real.min(), train_true_real.max()],
                           [train_true_real.min(), train_true_real.max()], 'r--', lw=2)
            axes[0, 3].set_xlabel('True Real Values')
            axes[0, 3].set_ylabel('Predicted Real Values')
            axes[0, 3].set_title(f'Training Set - Real Correlation: {correlation_real:.6f}')
            axes[0, 3].grid(True, alpha=0.3)

            # 测试集可视化
            test_length = test_lengths[0].item()
            test_coords = test_trunk[0, :test_length, :2].cpu().numpy()
            test_true_real = test_y[0, :test_length, 0].cpu().numpy()  # 实部
            test_true_imag = test_y[0, :test_length, 1].cpu().numpy()  # 虚部

            # 使用正确的长度进行预测（单Branch格式）
            test_trunk_single = test_trunk[0, :test_length].unsqueeze(0)
            test_pred_single = self._predict_single_sample(test_branch_input, test_trunk_single)
            test_pred_real = test_pred_single[:, 0]  # 预测实部
            test_pred_imag = test_pred_single[:, 1]  # 预测虚部

            test_error_real = np.abs(test_true_real - test_pred_real)
            test_error_imag = np.abs(test_true_imag - test_pred_imag)

            # 从branch数据中提取探针位置（单Branch格式：[x1,y1,r1,i1,...,freq]）
            test_branch_data = test_branch_input[0].cpu().numpy()
            # 计算实际的探针数量：(branch数据长度-1)/4，最后1位是频率
            actual_num_probes_test = (len(test_branch_data) - 1) // 4
            # 提取x,y坐标：每4个元素取前2个
            test_probe_coords = []
            for i in range(actual_num_probes_test):
                test_probe_coords.append([test_branch_data[i*4], test_branch_data[i*4+1]])
            test_probe_coords = np.array(test_probe_coords)  # [num_probes, 2]

            # 测试集 - 真实值实部 (带colorbar)
            scatter4 = axes[1, 0].scatter(test_coords[:, 0], test_coords[:, 1],
                                        c=test_true_real, cmap='viridis', s=5, alpha=0.6)
            axes[1, 0].set_title('Test Set - True Real Values')
            axes[1, 0].set_xlabel('X')
            axes[1, 0].set_ylabel('Y')
            plt.colorbar(scatter4, ax=axes[1, 0], label='真实场强(实部)')

            self._add_probe_markers(axes[1, 0], test_probe_coords)  # 默认显示所有探针
            self._add_convex_boundary(axes[1, 0], test_coords, alpha=0.3, color='blue', label='数据边界')
            self._add_convex_boundary(axes[1, 0], test_probe_coords, alpha=0.5, color='green', label='探针边界')

            # 测试集 - 预测值实部 (带colorbar)
            scatter5 = axes[1, 1].scatter(test_coords[:, 0], test_coords[:, 1],
                                        c=test_pred_real, cmap='viridis', s=5, alpha=0.6)
            axes[1, 1].set_title('Test Set - Predicted Real Values')
            axes[1, 1].set_xlabel('X')
            axes[1, 1].set_ylabel('Y')
            plt.colorbar(scatter5, ax=axes[1, 1], label='预测场强(实部)')

            self._add_probe_markers(axes[1, 1], test_probe_coords)  # 默认显示所有探针
            self._add_convex_boundary(axes[1, 1], test_coords, alpha=0.3, color='blue', label='数据边界')
            self._add_convex_boundary(axes[1, 1], test_probe_coords, alpha=0.5, color='green', label='探针边界')

            # 测试集 - 误差实部 (带colorbar)
            scatter6 = axes[1, 2].scatter(test_coords[:, 0], test_coords[:, 1],
                                        c=test_error_real, cmap='Reds', s=5, alpha=0.6)
            axes[1, 2].set_title(f'Test Set - Real Error (MAE: {np.mean(test_error_real):.6f})')
            axes[1, 2].set_xlabel('X')
            axes[1, 2].set_ylabel('Y')
            plt.colorbar(scatter6, ax=axes[1, 2], label='实部误差')

            # 测试集 - 实部相关性散点图
            correlation_test_real = np.corrcoef(test_true_real, test_pred_real)[0, 1]
            axes[1, 3].scatter(test_true_real, test_pred_real, alpha=0.6, s=2)
            axes[1, 3].plot([test_true_real.min(), test_true_real.max()],
                           [test_true_real.min(), test_true_real.max()], 'r--', lw=2)
            axes[1, 3].set_xlabel('True Real Values')
            axes[1, 3].set_ylabel('Predicted Real Values')
            axes[1, 3].set_title(f'Test Set - Real Correlation: {correlation_test_real:.6f}')
            axes[1, 3].grid(True, alpha=0.3)

            # 添加统计信息到相关性图
            correlation_text = (
                f"训练集统计:\n"
                f"实部相关性: {correlation_real:.6f}\n"
                f"实部MAE: {np.mean(train_error_real):.6f}\n"
                f"实部最大误差: {np.max(train_error_real):.6f}\n\n"
                f"测试集统计:\n"
                f"实部相关性: {correlation_test_real:.6f}\n"
                f"实部MAE: {np.mean(test_error_real):.6f}\n"
                f"实部最大误差: {np.max(test_error_real):.6f}"
            )

            # 在右下角图中添加统计信息
            axes[1, 3].text(0.02, 0.98, correlation_text, transform=axes[1, 3].transAxes,
                           fontsize=8, verticalalignment='top',
                           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

            plt.tight_layout()

            # 保存图片
            save_path = Path(self.cfg.visualization.output_dir) / "training_progress"
            save_path.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path / f'epoch_{epoch:04d}.png', dpi=150, bbox_inches='tight')
            plt.close()

            print(f"已保存训练进度图片: epoch_{epoch:04d}.png (2x4布局)")

    def plot_loss_curves(self, loss_history):
        """绘制loss曲线"""
        try:
            import matplotlib.pyplot as plt
            import matplotlib
            matplotlib.use('Agg')  # 使用非交互式后端

            # 设置中文字体
            plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
            plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
            try:
                matplotlib.font_manager.fontManager.addfont(
                    matplotlib.font_manager.FontProperties(family='SimHei')
                )
            except:
                pass  # 如果字体不存在，跳过

            epochs = range(1, len(loss_history['train_total']) + 1)

            # 创建2x2子图
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))

            # 1. 总loss对比
            axes[0, 0].plot(epochs, loss_history['train_total'], 'b-', label='训练总loss', linewidth=2)
            axes[0, 0].plot(epochs, loss_history['test_total'], 'r-', label='测试总loss', linewidth=2)
            axes[0, 0].set_title('总loss变化')
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)

            # 2. 各项训练loss
            axes[0, 1].plot(epochs, loss_history['train_probe'], 'g-', label='探针loss', linewidth=2)
            axes[0, 1].plot(epochs, loss_history['train_field'], 'orange', label='场loss', linewidth=2)
            axes[0, 1].plot(epochs, loss_history['train_correlation'], 'purple', label='相关性loss', linewidth=2)
            if loss_history['train_smooth'][-1] > 0:
                axes[0, 1].plot(epochs, loss_history['train_smooth'], 'brown', label='平滑loss', linewidth=2)
            axes[0, 1].set_title('各项训练loss变化')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Loss')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)

            # 3. 训练vs测试总loss (对数坐标)
            axes[1, 0].semilogy(epochs, loss_history['train_total'], 'b-', label='训练总loss', linewidth=2)
            axes[1, 0].semilogy(epochs, loss_history['test_total'], 'r-', label='测试总loss', linewidth=2)
            axes[1, 0].set_title('总loss变化 (对数坐标)')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Loss (log scale)')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)

            # 4. 损失占比分析
            train_probe_contrib = [self.cfg.physics.probe_loss_weight * p for p in loss_history['train_probe']]
            train_field_contrib = [self.cfg.physics.field_loss_weight * f for f in loss_history['train_field']]
            train_correlation_contrib = [self.cfg.physics.correlation_loss_weight * c for c in loss_history['train_correlation']]

            axes[1, 1].stackplot(epochs, train_probe_contrib, train_field_contrib, train_correlation_contrib,
                                 labels=['探针贡献', '场贡献', '相关性贡献'], alpha=0.7)
            axes[1, 1].set_title('损失组成占比')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Loss Contribution')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)

            plt.tight_layout()

            # 保存图片
            save_path = Path(self.cfg.visualization.output_dir) / "loss_curves"
            save_path.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path / 'loss_curves.png', dpi=300, bbox_inches='tight')
            plt.close()

            print(f"已保存loss曲线图: {save_path}/loss_curves.png")

            # 输出最佳结果
            best_train_idx = np.argmin(loss_history['train_total'])
            best_test_idx = np.argmin(loss_history['test_total'])

            print(f"\n=== 训练结果总结 ===")
            print(f"最佳训练loss: {loss_history['train_total'][best_train_idx]:.6f} (Epoch {best_train_idx + 1})")
            print(f"最佳测试loss: {loss_history['test_total'][best_test_idx]:.6f} (Epoch {best_test_idx + 1})")
            print(f"最终训练loss: {loss_history['train_total'][-1]:.6f}")
            print(f"最终测试loss: {loss_history['test_total'][-1]:.6f}")

        except Exception as e:
            print(f"绘制loss曲线失败: {e}")

    def _predict_single_sample(self, branch_input, trunk):
        """预测单个样本 - 单Branch版本返回实部和虚部"""
        # 期望的branch维度：25 probes × 4 features + 1 freq = 101
        expected_branch_size = self.cfg.deeponet.probe_count * 4 + 1

        # 重构branch_input维度
        if branch_input.dim() == 2:
            branch_flat = branch_input[0].view(-1)
        else:
            branch_flat = branch_input.view(-1)

        # 确保branch_flat是正确的大小 [101]
        if branch_flat.numel() != expected_branch_size:
            if branch_flat.numel() > expected_branch_size:
                branch_flat = branch_flat[:expected_branch_size]
            else:
                padding = torch.zeros(expected_branch_size - branch_flat.numel(), device=branch_flat.device)
                branch_flat = torch.cat([branch_flat, padding])

        # 重构trunk维度
        if trunk.dim() == 3:
            trunk = trunk.squeeze(0)
            seq_length = trunk.shape[0]
        elif trunk.dim() == 2:
            seq_length = trunk.shape[0]
        else:
            seq_length = 1000

        # 准备模型输入 - 保持正确的形状
        branch_input_shaped = branch_flat.unsqueeze(0)  # [1, 101]
        trunk_input = trunk.unsqueeze(0) if trunk.dim() == 2 else trunk.unsqueeze(0)  # [1, seq_length, 3]

        # 模型预测（单Branch输入）
        try:
            with torch.no_grad():
                predictions = self.model(branch_input_shaped, trunk_input)
                return predictions[0].cpu().numpy()  # 返回 [seq_length, 2]

        except Exception as e:
            print(f"[DEBUG] Prediction error: {e}")
            # 返回零数组作为fallback
            return np.zeros((seq_length, 2))

    def _add_probe_markers(self, ax, probe_coords, step=1):
        """添加探针位置标注"""
        for i, (px, py) in enumerate(probe_coords[::step]):  # 每step个显示一个探针
            ax.scatter(px, py, c='red', s=20, marker='^', alpha=0.8,
                      edgecolors='white', linewidth=1)
            if i < 10:  # 只显示前10个的标签
                ax.annotate(f'P{i*step}', (px, py), xytext=(2, 2),
                          textcoords='offset points', fontsize=6, color='red')

    def _add_convex_boundary(self, ax, coords, alpha=0.3, color='blue', label=None):
        """添加凸边界显示"""
        try:
            from scipy.spatial import ConvexHull
            import numpy as np

            # 计算凸包
            hull = ConvexHull(coords)

            # 绘制凸包边界线
            boundary_line = None
            for simplex in hull.simplices:
                line = ax.plot(coords[simplex, 0], coords[simplex, 1],
                             color=color, alpha=alpha, linewidth=2)
                if boundary_line is None:
                    boundary_line = line[0]

            # 填充凸包区域
            ax.fill(coords[hull.vertices, 0], coords[hull.vertices, 1],
                   color=color, alpha=alpha*0.3, label=label if label else None)

        except Exception as e:
            # 如果scipy不可用或计算失败，绘制简单的边界框
            print(f"凸包计算失败，使用边界框: {e}")
            x_min, x_max = np.min(coords[:, 0]), np.max(coords[:, 0])
            y_min, y_max = np.min(coords[:, 1]), np.max(coords[:, 1])

            # 绘制边界框
            from matplotlib.patches import Rectangle
            rect = Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                           linewidth=2, edgecolor=color, facecolor=color,
                           alpha=alpha*0.3, label=label if label else None)
            ax.add_patch(rect)

    def _predict_single_sample(self, branch, trunk):
        """预测单个样本"""
        # 期望的branch维度
        expected_branch_size = self.cfg.deeponet.probe_count * 5

        # 重构branch维度
        if branch.dim() == 3:
            # [1, num_probes, 5] -> [probe_count * 5]
            branch_flat = branch.squeeze(0).view(-1)
        elif branch.dim() == 2:
            # [batch_size, probe_count * 5] -> 只取第一个样本 [probe_count * 5]
            branch_flat = branch[0].view(-1)
        else:
            # [probe_count * 5] 直接使用
            branch_flat = branch.view(-1)

        # 确保branch_flat是正确的大小 [probe_count * 5]
        if branch_flat.numel() != expected_branch_size:
            if branch_flat.numel() == 50 * 5:  # 旧的50探针数据
                # 如果是250但期望更大，需要重复填充
                if expected_branch_size > 250:
                    repeats = expected_branch_size // 250
                    remainder = expected_branch_size % 250
                    branch_flat = branch_flat.repeat(repeats)
                    if remainder > 0:
                        branch_flat = torch.cat([branch_flat, branch_flat[:remainder]])
                else:
                    branch_flat = branch_flat.view(-1)
            else:
                print(f"[DEBUG] Unexpected branch shape in _predict_single_sample: {branch_flat.shape}")
                # 如果元素不对，截断或填充到expected_branch_size
                if branch_flat.numel() > expected_branch_size:
                    branch_flat = branch_flat[:expected_branch_size]
                else:
                    padding = torch.zeros(expected_branch_size - branch_flat.numel(), device=branch_flat.device)
                    branch_flat = torch.cat([branch_flat, padding])

        # 重构trunk维度
        if trunk.dim() == 3:
            # [1, N, 3] -> [N, 3]
            trunk = trunk.squeeze(0)
            seq_length = trunk.shape[0]
        elif trunk.dim() == 2:
            seq_length = trunk.shape[0]
        else:
            print(f"[DEBUG] Unexpected trunk shape: {trunk.shape}")
            seq_length = 1000  # 默认长度

        # 准备模型输入 - 保持正确的形状，不要错误扩展
        branch_input = branch_flat.unsqueeze(0)  # [1, probe_count * 5]
        trunk_input = trunk.unsqueeze(0)  # [1, seq_length, 3]

        # 模型预测
        try:
            with torch.no_grad():
                predictions = self.model(branch_input, trunk_input)
                return predictions[0].cpu().numpy()  # 返回 [seq_length, 2]

        except Exception as e:
            print(f"[DEBUG] Prediction error in _predict_single_sample: {e}")
            # 返回零数组作为fallback
            return np.zeros((seq_length, 2))

    def evaluate_optimized(self, dataloader) -> float:
        """优化版评估函数 - 真正的批量处理"""
        self.model.eval()

        test_losses = {'total': 0, 'probe': 0, 'field': 0, 'smooth': 0}
        num_batches = 0

        with torch.no_grad():
            for batch in dataloader:
                # 移动数据到设备 - 单Branch格式
                branch_input_batch = batch['branch_input'].to(self.device)
                trunk_batch = batch['trunk'].to(self.device)
                y_batch = batch['y'].to(self.device)
                mask_batch = batch['mask'].to(self.device)
                lengths = batch['lengths'].to(self.device)
                probe_coords = batch['probe_coords'].to(self.device)
                sample_ids = batch['sample_ids'].to(self.device)

                # 获取真实探针值
                probe_true_values = batch.get('probe_true_values', None)
                if probe_true_values is not None:
                    probe_true_values = probe_true_values.to(self.device)

                current_batch_size = branch_input_batch.shape[0]

                # 批量预测 - 单Branch输入，传递真实探针值用于RBF校正
                if probe_true_values is not None:
                    y_pred = self.model(branch_input_batch, trunk_batch,
                                      mask=mask_batch, lengths=lengths,
                                      probe_true_values=probe_true_values)
                else:
                    y_pred = self.model(branch_input_batch, trunk_batch,
                                      mask=mask_batch, lengths=lengths)

                # 批量损失计算 - 传递coords和sample_ids用于Spectral Loss
                loss, loss_dict = self.loss_fn(
                    y_pred, y_batch,
                    [mask_batch[i] for i in range(mask_batch.shape[0])],
                    lengths,
                    coords=trunk_batch,
                    sample_ids=sample_ids
                )

                # 记录
                test_losses['total'] += loss.item()

                # 安全地提取损失值（处理tensor和float类型）
                if hasattr(loss_dict['probe_loss'], 'item'):
                    test_losses['probe'] += loss_dict['probe_loss'].item()
                else:
                    test_losses['probe'] += float(loss_dict['probe_loss'])

                if hasattr(loss_dict['field_loss'], 'item'):
                    test_losses['field'] += loss_dict['field_loss'].item()
                else:
                    test_losses['field'] += float(loss_dict['field_loss'])

                if hasattr(loss_dict['smooth_loss'], 'item'):
                    test_losses['smooth'] += loss_dict['smooth_loss'].item()
                else:
                    test_losses['smooth'] += float(loss_dict['smooth_loss'])

                num_batches += 1

        # 平均损失
        if num_batches > 0:
            for key in test_losses:
                test_losses[key] /= num_batches
        else:
            print(f"[WARNING] No batches processed in evaluation")
            return 0.0

        return test_losses['total']

    def train_optimized(self, train_loader, test_loader):
        """优化版训练主循环 - 真正的批量处理"""
        print(f"\n=== 开始优化版训练 ===")
        print(f"训练轮数: {self.cfg.training.epochs}")
        print(f"关键优化：移除逐点处理和逐样本处理for循环")

        # 打印关键训练配置
        print(f"\n=== 训练配置确认 ===")
        print(f"数据路径: {self.cfg.data.data_path}")
        print(f"探针数量: {self.cfg.data.num_probes}")
        print(f"批次大小: {self.cfg.training.batch_size}")
        print(f"学习率: {self.cfg.training.learning_rate}")
        print(f"网络预设: {self.cfg.deeponet.network_preset}")
        print(f"Dropout: {self.cfg.deeponet.dropout_rate}")
        print(f"注意力: {'启用' if self.cfg.deeponet.use_attention else '禁用'}")
        print(f"残差: {'启用' if self.cfg.deeponet.use_residual else '禁用'}")
        print(f"RBF校正: {'启用' if self.cfg.physics.enable_rbf_correction else '禁用'}")
        print("=" * 60)

        best_test_loss = float('inf')
        patience_counter = 0
        train_losses = []
        test_losses = []

        # 详细的loss跟踪（包含Spectral Loss）
        loss_history = {
            'train_total': [],
            'test_total': [],
            'train_probe': [],
            'train_field': [],
            'train_correlation': [],
            'train_smooth': [],
            'train_spectral': []  # Day 1核心：Spectral Loss跟踪
        }

        for epoch in range(self.cfg.training.epochs):
            epoch_start_time = time.time()

            # 训练
            train_loss, train_loss_details = self.train_epoch_optimized(train_loader, epoch + 1)

            # 评估
            test_loss = self.evaluate_optimized(test_loader)

            # 记录损失历史（包含Spectral Loss）
            loss_history['train_total'].append(train_loss)
            loss_history['test_total'].append(test_loss)
            loss_history['train_probe'].append(train_loss_details['probe'])
            loss_history['train_field'].append(train_loss_details['field'])
            loss_history['train_correlation'].append(train_loss_details['correlation'])
            loss_history['train_smooth'].append(train_loss_details['smooth'])
            loss_history['train_spectral'].append(train_loss_details.get('spectral_loss', 0.0))  # Day 1核心

            # TensorBoard日志记录
            self.writer.add_scalar("Loss/Train_Total", train_loss, epoch)
            self.writer.add_scalar("Loss/Test_Total", test_loss, epoch)
            self.writer.add_scalar("Loss/Train_Probe", train_loss_details['probe'], epoch)
            self.writer.add_scalar("Loss/Train_Field", train_loss_details['field'], epoch)
            self.writer.add_scalar("Loss/Train_Correlation", train_loss_details['correlation'], epoch)
            self.writer.add_scalar("Loss/Train_Smooth", train_loss_details['smooth'], epoch)
            self.writer.add_scalar("Loss/Train_Spectral", train_loss_details.get('spectral_loss', 0.0), epoch)  # Day 1核心

            # 学习率日志
            current_lr = self.optimizer.param_groups[0]['lr']
            self.writer.add_scalar("Learning_Rate", current_lr, epoch)

            # 参数梯度范数（可选）
            total_norm = 0
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    param_norm = param.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** (1. / 2)
            self.writer.add_scalar("Gradients/Total_Norm", total_norm, epoch)

            self.writer.flush()  # 立即写入

            # 学习率调度
            self.scheduler.step(test_loss)
            current_lr = self.optimizer.param_groups[0]['lr']

            # 记录损失
            train_losses.append(train_loss)
            test_losses.append(test_loss)

            epoch_time = time.time() - epoch_start_time

            # 计算Pure MSE (Data Only, 不含Spectral Loss)
            pure_mse_train = (train_loss_details['probe'] +
                             train_loss_details['field'] +
                             train_loss_details['correlation'] +
                             train_loss_details['smooth'])
            spectral_train = train_loss_details.get('spectral_loss', 0.0)

            # 每个epoch都显示详细Loss拆分
            print(f"[Epoch {epoch + 1}/{self.cfg.training.epochs}] "
                  f"Total: {train_loss:.4f} | "
                  f"Pure MSE (Data): {pure_mse_train:.4f} | "
                  f"Spectral: {spectral_train:.4f} | "
                  f"Test: {test_loss:.4f}")

            # 打印详细进度（每display_every个epoch）
            if (epoch + 1) % self.cfg.training.display_every == 0:
                print(f"Epoch {epoch + 1}/{self.cfg.training.epochs} 详细拆分:")
                print(f"  训练 Total Loss: {train_loss:.6f}")
                print(f"  训练 Pure MSE (Data Only): {pure_mse_train:.6f}")
                print(f"    ├─ Probe: {train_loss_details['probe']:.6f}")
                print(f"    ├─ Field: {train_loss_details['field']:.6f}")
                print(f"    ├─ Correlation: {train_loss_details['correlation']:.6f}")
                print(f"    └─ Smooth: {train_loss_details['smooth']:.6f}")
                print(f"  训练 Spectral Loss (k-space): {spectral_train:.6f}")
                print(f"  测试 Total Loss: {test_loss:.6f}")
                print(f"  归一化状态: normalize_data=False (原始物理数值)")
                print("-" * 60)

            # 每plot_frequency个epoch保存可视化图片
            if (epoch + 1) % self.cfg.training.plot_frequency == 0:
                try:
                    self.save_training_visualizations(epoch + 1, train_loader, test_loader, train_loss, test_loss)
                except Exception as e:
                    # 如果可视化失败，简单跳过，不影响训练
                    pass
            else:
                # 跳过可视化保存，不影响训练
                pass

            # 早停和保存
            if test_loss < best_test_loss:
                best_test_loss = test_loss
                patience_counter = 0

                # 只在测试损失创新低时保存最优模型
                print(f"[NEW BEST] 最佳测试损失: {best_test_loss:.6f} (Epoch {epoch + 1})")
                save_checkpoint(
                    self.model, self.optimizer,
                    epoch=epoch + 1,
                    loss_history=train_losses,
                    cfg=self.cfg,
                    save_dir=self.cfg.data.checkpoint_dir,
                    model_type="pytorch_optimized",
                    is_best_model=True  # 标记为最佳模型
                )
            else:
                patience_counter += 1
                if patience_counter >= self.cfg.training.early_stopping_patience:
                    print(f"早停触发，在Epoch {epoch + 1}")
                    break

        total_time = time.time() - epoch_start_time
        print(f"优化版训练完成！总用时: {total_time/60:.1f} 分钟")
        print(f"最佳测试损失: {best_test_loss:.6f}")

        # 绘制loss曲线
        self.plot_loss_curves(loss_history)

        # 关闭TensorBoard writer
        if hasattr(self, 'writer'):
            self.writer.close()
            print(f"TensorBoard日志已关闭")

        # 保存最终模型
        final_epoch = len(train_losses)
        print(f"\n[TRAINING COMPLETE] 保存最终模型 (Epoch {final_epoch})")
        save_checkpoint(
            self.model, self.optimizer,
            epoch=final_epoch,
            loss_history=train_losses,
            cfg=self.cfg,
            save_dir=self.cfg.data.checkpoint_dir,
            model_type="pytorch_optimized",
            is_final=True  # 标记为最终模型
        )

        return train_losses, test_losses


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='修复版DeepONet电磁场重构训练')

    # 配置文件参数
    parser.add_argument('--config_file', type=str, default='config.yaml',
                       help='YAML配置文件路径')
    parser.add_argument('--preset', type=str, default=None,
                       help='使用的预设配置名称')

    # 数据参数
    parser.add_argument('--data_path', type=str,
                       default='',  # 将从配置文件读取，如果未指定则使用默认配置
                       help='数据文件路径 (默认从config.yaml读取)')
    parser.add_argument('--max_samples', type=int, default=None,
                       help='最大样本数 (None表示使用所有样本)')
    parser.add_argument('--num_probes', type=int, default=None,
                       help='探针数量（默认从config.yaml读取）')

    # 训练参数
    parser.add_argument('--epochs', type=int, default=20,
                       help='训练轮数')
    parser.add_argument('--lr', type=float, default=5e-4,
                       help='学习率')
    parser.add_argument('--batch_size', type=int, default=None,
                       help='批次大小（默认从配置文件读取）')

    # 输出参数
    parser.add_argument('--output_dir', type=str, default=None,
                       help='输出目录（默认从配置文件读取）')
    parser.add_argument('--checkpoint_dir', type=str, default=None,
                       help='检查点目录（默认从配置文件读取）')

    # 显示参数
    parser.add_argument('--display_every', type=int, default=2,
                       help='显示频率')
    parser.add_argument('--save_interval', type=int, default=5,
                       help='保存间隔')
    parser.add_argument('--plot_frequency', type=int, default=2,
                       help='图片保存频率')

    # 恢复训练
    parser.add_argument('--resume', type=str, default=None,
                       help='恢复训练的检查点路径')

    return parser.parse_args()


def main():
    """主函数"""
    print("=== 修复版DeepONet电磁场重构训练 ===")
    print("修复变长序列和ReduceLROnPlateau问题")
    print("支持YAML配置文件和凸边界筛选")
    print("=" * 60)

    # 解析参数
    args = parse_args()

    # 从YAML配置文件加载配置
    config_file = args.config_file if Path(args.config_file).exists() else None
    cfg = Config(config_file=config_file, preset_name=args.preset)

    # 命令行参数覆盖配置文件
    if args.data_path:
        cfg.data.data_path = args.data_path
    elif hasattr(cfg, 'paths') and not cfg.data.data_path:
        # 如果命令行没有指定且data为空，从paths读取
        cfg.data.data_path = cfg.paths.data_path

    if args.max_samples:
        cfg.data.max_samples = args.max_samples
    if hasattr(args, 'num_probes') and args.num_probes:
        cfg.data.num_probes = args.num_probes
        # 关键修复：同步更新deeponet配置中的探针数量和输入维度
        cfg.sync_probe_count()
        print(f"[排查] 探针数量配置覆盖: data.num_probes={cfg.data.num_probes}, deeponet.probe_count={cfg.deeponet.probe_count}, deeponet.input_dim={cfg.deeponet.input_dim}")

    cfg.training.epochs = args.epochs
    if args.lr:
        cfg.training.learning_rate = args.lr
    if args.batch_size is not None:
        cfg.training.batch_size = args.batch_size
    cfg.training.display_every = args.display_every
    cfg.training.save_interval = args.save_interval

    if hasattr(args, 'plot_frequency') and args.plot_frequency:
        cfg.training.plot_frequency = args.plot_frequency

    if args.output_dir:
        cfg.visualization.output_dir = args.output_dir
    if args.checkpoint_dir:
        cfg.data.checkpoint_dir = args.checkpoint_dir

    # 创建输出目录
    Path(cfg.visualization.output_dir).mkdir(exist_ok=True)
    Path(cfg.data.checkpoint_dir).mkdir(exist_ok=True)

    # 打印配置
    cfg.print_config()

    try:
        # 准备数据
        print("准备修复版数据集...")
        dataset = MaskedDeepONetDataset(cfg)

        sample_files = dataset.scan_available_samples(
            cfg.data.data_path,
            max_frequency=cfg.data.max_frequency,
            max_samples=cfg.data.max_samples
        )

        if not sample_files:
            raise ValueError("未找到有效的数据文件")

        print(f"总共发现 {len(sample_files)} 个可用构型")

        # 限制样本数量
        if cfg.data.max_samples and len(sample_files) > cfg.data.max_samples:
            sample_files = sample_files[:cfg.data.max_samples]
            print(f"限制样本数量为: {len(sample_files)}")

        # 随机打乱样本
        np.random.shuffle(sample_files)

        # 划分训练集和测试集索引
        print(f"[DEBUG] Total sample files: {len(sample_files)}")

        # 当样本数量很少时，使用不同的划分策略
        if len(sample_files) <= 2:
            # 样本太少时，至少保证每个集合有1个样本
            if len(sample_files) == 1:
                train_samples = sample_files[:1]  # 唯一样本给训练集
                test_samples = sample_files[:1]   # 测试集也用同一个样本
                print("[INFO] 样本数量很少，训练集和测试集使用同一个样本")
            elif len(sample_files) == 2:
                train_samples = sample_files[:1]  # 第一个样本给训练集
                test_samples = sample_files[1:]   # 第二个样本给测试集
                print("[INFO] 使用1:1划分训练集和测试集")
        else:
            # 正常的80%/20%划分
            split_idx = int(len(sample_files) * 0.8)
            if split_idx == 0:
                split_idx = 1  # 确保训练集至少有1个样本
            if split_idx == len(sample_files):
                split_idx = len(sample_files) - 1  # 确保测试集至少有1个样本

            train_samples = sample_files[:split_idx]
            test_samples = sample_files[split_idx:]
            print(f"[INFO] 使用标准80%/20%划分，split_idx: {split_idx}")

        print(f"训练集构型数: {len(train_samples)}")
        print(f"测试集构型数: {len(test_samples)}")

        # 创建可动态加载的数据集
        train_dataset = DynamicDeepONetDataset(train_samples, cfg)
        test_dataset = DynamicDeepONetDataset(test_samples, cfg)

        # 调试选项：固定探针选择以测试探针loss收敛
        # 设置为True可以消除探针选择噪声，帮助验证探针loss是否能快速下降
        train_dataset._fixed_probe_mode = False  # 可以改为True进行调试
        test_dataset._fixed_probe_mode = False   # 测试集保持随机

        print(f"[排查] 数据集探针配置验证:")
        print(f"  train_dataset.dataset.probe_count={train_dataset.dataset.probe_count}")
        print(f"  cfg.data.num_probes={cfg.data.num_probes}")
        print(f"  cfg.deeponet.probe_count={cfg.deeponet.probe_count}")
        print(f"  预期branch输入维度={cfg.data.num_probes * 5}")
        print(f"  固定探针模式={train_dataset._fixed_probe_mode}")

        # 创建数据加载器（使用自定义collate函数）
        from torch.utils.data import SubsetRandomSampler, SequentialSampler
        import random

        # 方案1：使用SubsetRandomSampler进行真正的随机采样
        train_indices = list(range(len(train_dataset)))
        train_sampler = SubsetRandomSampler(train_indices)  # ✅ 真正的随机采样

        # 使用较少的workers以避免CUDA multiprocess问题
        num_workers = 0  # 暂时禁用多进程避免CUDA冲突，确保训练稳定运行

        print(f"使用 {num_workers} 个数据加载进程")
        print(f"训练集使用SubsetRandomSampler进行真正的随机采样")

        # 创建lambda函数来传递配置
        train_collate_fn = lambda batch: collate_fn(batch, cfg)
        test_collate_fn = lambda batch: collate_fn(batch, cfg)

        train_loader = DataLoader(
            train_dataset,
            batch_size=cfg.training.batch_size,
            sampler=train_sampler,  # 使用SubsetRandomSampler
            num_workers=num_workers,  # 启用多进程数据加载
            pin_memory=False,  # 禁用pin_memory避免CUDA冲突
            collate_fn=train_collate_fn  # 关键：使用自定义collate函数
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=cfg.training.batch_size,
            sampler=SequentialSampler(range(len(test_dataset))),  # 测试集不需要shuffle
            num_workers=num_workers,  # 启用多进程数据加载
            pin_memory=False,  # 禁用pin_memory避免CUDA冲突
            collate_fn=test_collate_fn  # 关键：使用自定义collate函数
        )

        print(f"训练批次数: {len(train_loader)}")
        print(f"测试批次数: {len(test_loader)}")

        # 验证数据加载器不为空
        if len(train_loader) == 0:
            raise ValueError("训练数据加载器为空！")
        if len(test_loader) == 0:
            raise ValueError("测试数据加载器为空！")

        # 创建训练器
        trainer = ImprovedDeepONetTrainer(cfg)

        # 恢复训练（如果指定）
        if args.resume:
            try:
                epoch, loss_history, config_dict = load_checkpoint(
                    args.resume, trainer.model, trainer.optimizer, trainer.device
                )
                print(f"从检查点恢复训练: Epoch {epoch}")
            except Exception as e:
                print(f"恢复训练失败: {e}")
                return

        # 开始优化版训练
        train_losses, test_losses = trainer.train_optimized(train_loader, test_loader)

        print("\n优化版训练成功完成！")
        print(f"结果保存在: {cfg.visualization.output_dir}")
        print(f"检查点保存在: {cfg.data.checkpoint_dir}")

        # 输出性能提升总结
        print("\n=== 性能优化总结 ===")
        print("1. 修复模型逐点处理bug: O(N) → O(1) 常数时间")
        print("2. 移除训练循环中的样本级for循环: 真正的批量处理")
        print("3. 充分利用GPU并行计算能力")
        print("4. 预期性能提升: 10-100倍加速")
        print("==================")

    except Exception as e:
        print(f"修复版训练失败: {e}")
        raise


if __name__ == "__main__":
    main()