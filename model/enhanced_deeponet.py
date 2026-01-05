"""
增强的DeepONet模型定义
用于电磁场重构的单Branch DeepONet架构，支持残差连接、注意力机制、Dropout等增强功能

架构特性：
- 单Branch架构：统一处理实部和虚部，捕捉相位耦合
- 支持多种增强功能：残差连接、注意力、Dropout
- 模块化设计：根据配置自动选择合适的网络复杂度
- 批量处理：支持高效的批量训练和推理

使用示例：
    model = create_enhanced_deeponet(cfg)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
from pathlib import Path

# 添加父目录到路径
sys.path.append(str(Path(__file__).parent.parent))

from config.config import Config
from .enhanced_layers import EnhancedMLPBlock, FusionMLP
from .probe_correction import create_probe_correction_module


class SingleBranchDeepONet(nn.Module):
    """单Branch DeepONet - 统一处理复数场的实部和虚部"""

    def __init__(self, cfg: Config):
        super(SingleBranchDeepONet, self).__init__()
        self.cfg = cfg

        # 获取网络配置
        hidden_layers = cfg.deeponet.hidden_layers
        output_dim = cfg.deeponet.output_dim
        branch_input_dim = cfg.deeponet.input_dim  # 25*4+1 = 101

        print(f"=== 初始化单Branch DeepONet ===")
        print(f"网络预设: {cfg.deeponet.network_preset}")
        print(f"Branch输入维度: {branch_input_dim}")
        print(f"隐藏层: {hidden_layers}")
        print(f"输出维度: {output_dim}")
        print(f"激活函数: {cfg.deeponet.activation}")
        print(f"Dropout率: {cfg.deeponet.dropout_rate}")
        print(f"使用残差连接: {cfg.deeponet.use_residual}")
        print(f"使用注意力机制: {cfg.deeponet.use_attention}")
        print("=" * 40)

        # 构建Branch网络（处理探针数据）
        self.branch_layers = nn.ModuleList()
        prev_dim = branch_input_dim
        for hidden_dim in hidden_layers:
            self.branch_layers.append(
                EnhancedMLPBlock(prev_dim, hidden_dim, cfg.deeponet)
            )
            prev_dim = hidden_dim

        # Branch输出层
        self.branch_output = nn.Linear(prev_dim, output_dim)

        # 构建Trunk网络（处理空间坐标）
        self.trunk_layers = nn.ModuleList()
        trunk_input_dim = 3  # [x, y, frequency]

        prev_dim = trunk_input_dim
        for hidden_dim in hidden_layers:
            self.trunk_layers.append(
                EnhancedMLPBlock(prev_dim, hidden_dim, cfg.deeponet)
            )
            prev_dim = hidden_dim

        # Trunk输出层
        self.trunk_output = nn.Linear(prev_dim, output_dim)

        # 融合层（单一融合MLP）
        self.fusion = FusionMLP(
            branch_dim=output_dim,
            trunk_dim=output_dim,
            hidden_dims=[output_dim*2, output_dim],  # 两层融合MLP
            output_dim=output_dim,
            activation=cfg.deeponet.activation
        )

        # 双输出头：分别输出实部和虚部
        self.output_real = nn.Linear(output_dim, 1)
        self.output_imag = nn.Linear(output_dim, 1)

        # 初始化探针校正模块（Day 1暂时禁用）
        # self.probe_correction = create_probe_correction_module(cfg)

        # 网络权重初始化
        self._initialize_weights()

    def _initialize_weights(self):
        """初始化网络权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                if self.cfg.deeponet.initializer == "Glorot normal":
                    nn.init.xavier_normal_(module.weight)
                elif self.cfg.deeponet.initializer == "He normal":
                    nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
                else:
                    nn.init.normal_(module.weight, 0, 0.02)

                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, branch_input, trunk_input, mask=None, lengths=None,
                return_correction_info=False, probe_true_values=None):
        """
        单Branch前向传播

        Args:
            branch_input: [batch_size, 101] 单Branch探针数据
            trunk_input: [batch_size, N, 3] 空间坐标数据
            mask: [batch_size, N] 探针位置mask (可选)
            lengths: [batch_size] 有效长度 (可选)
            return_correction_info: bool 是否返回校正信息
            probe_true_values: [batch_size, probe_count, 2] 真实探针值 (可选)

        Returns:
            torch.Tensor: [batch_size, N, 2] (real, imag)
        """
        # 处理输入维度
        if branch_input.dim() == 1:
            branch_input = branch_input.unsqueeze(0)
            single_sample = True
        else:
            single_sample = False

        if trunk_input.dim() == 2:
            trunk_input = trunk_input.unsqueeze(0)

        batch_size = trunk_input.shape[0]
        num_points = trunk_input.shape[1]

        # Branch网络前向传播
        branch_out = branch_input
        for layer in self.branch_layers:
            branch_out = layer(branch_out)
        branch_out = self.branch_output(branch_out)  # [batch_size, output_dim]

        # Trunk网络前向传播
        trunk_flat = trunk_input.view(-1, 3)  # [batch_size * num_points, 3]
        trunk_out = trunk_flat
        for layer in self.trunk_layers:
            trunk_out = layer(trunk_out)
        trunk_out = self.trunk_output(trunk_out)
        trunk_out = trunk_out.view(batch_size, num_points, -1)  # [batch_size, N, output_dim]

        # 扩展branch输出以匹配trunk的所有点
        branch_expanded = branch_out.unsqueeze(1).expand(-1, num_points, -1)  # [batch_size, N, output_dim]

        # 融合MLP
        fused = self.fusion(branch_expanded, trunk_out)  # [batch_size, N, output_dim]

        # 双输出头
        real_output = self.output_real(fused)  # [batch_size, N, 1]
        imag_output = self.output_imag(fused)  # [batch_size, N, 1]

        # 合并为最终输出 [batch_size, N, 2]
        output = torch.cat([real_output, imag_output], dim=-1)

        # 如果是单样本输入，去掉batch维度
        if single_sample:
            output = output.squeeze(0)  # [N, 2]

        return output

    def get_model_info(self):
        """获取模型信息"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        info = {
            'model_type': 'SingleBranchDeepONet',
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'network_preset': self.cfg.deeponet.network_preset,
            'hidden_layers': self.cfg.deeponet.hidden_layers,
            'output_dim': self.cfg.deeponet.output_dim,
            'activation': self.cfg.deeponet.activation,
            'dropout_rate': self.cfg.deeponet.dropout_rate,
            'use_residual': self.cfg.deeponet.use_residual,
            'use_attention': self.cfg.deeponet.use_attention,
            'probe_count': self.cfg.deeponet.probe_count,
            'input_dim': self.cfg.deeponet.input_dim
        }

        return info

    def print_model_info(self):
        """打印模型信息"""
        info = self.get_model_info()
        print("\n" + "="*60)
        print(f"模型类型: {info['model_type']}")
        print(f"网络预设: {info['network_preset']}")
        print(f"隐藏层结构: {info['hidden_layers']}")
        print(f"特征维度: {info['output_dim']}")
        print(f"激活函数: {info['activation']}")
        print(f"Dropout率: {info['dropout_rate']}")
        print(f"残差连接: {'启用' if info['use_residual'] else '禁用'}")
        print(f"注意力机制: {'启用' if info['use_attention'] else '禁用'}")
        print(f"探针数量: {info['probe_count']}")
        print(f"输入维度: {info['input_dim']}")
        print("-"*60)
        print(f"总参数数量: {info['total_parameters']:,}")
        print(f"可训练参数: {info['trainable_parameters']:,}")
        print("="*60)


class EnhancedPyTorchDualBranchDeepONet(nn.Module):
    """增强版PyTorch双分支DeepONet - 支持残差、注意力、Dropout等高级功能"""

    def __init__(self, cfg: Config):
        super(EnhancedPyTorchDualBranchDeepONet, self).__init__()
        self.cfg = cfg

        # 获取网络配置
        hidden_layers = cfg.deeponet.hidden_layers
        output_dim = cfg.deeponet.output_dim

        print(f"=== 初始化增强版DeepONet ===")
        print(f"网络预设: {cfg.deeponet.network_preset}")
        print(f"隐藏层: {hidden_layers}")
        print(f"激活函数: {cfg.deeponet.activation}")
        print(f"Dropout率: {cfg.deeponet.dropout_rate}")
        print(f"使用残差连接: {cfg.deeponet.use_residual}")
        print(f"使用注意力机制: {cfg.deeponet.use_attention}")
        print("=" * 40)

        # 构建Trunk网络（处理坐标，两个分支共享）
        self.trunk_layers = nn.ModuleList()
        trunk_input_dim = 3  # [x, y, z]

        prev_dim = trunk_input_dim
        for hidden_dim in hidden_layers:
            self.trunk_layers.append(
                EnhancedMLPBlock(prev_dim, hidden_dim, cfg.deeponet)
            )
            prev_dim = hidden_dim

        # Trunk输出层
        self.trunk_output = nn.Linear(prev_dim, output_dim)

        # 构建Branch网络（处理探针数据）
        branch_input_dim = cfg.deeponet.probe_count * 5  # [x,y,z,freq,measurement_value]

        # 实部Branch网络
        self.branch_real_layers = nn.ModuleList()
        prev_dim = branch_input_dim
        for hidden_dim in hidden_layers:
            self.branch_real_layers.append(
                EnhancedMLPBlock(prev_dim, hidden_dim, cfg.deeponet)
            )
            prev_dim = hidden_dim

        # 实部Branch输出层
        self.branch_real_output = nn.Linear(prev_dim, output_dim)

        # 虚部Branch网络
        self.branch_imag_layers = nn.ModuleList()
        prev_dim = branch_input_dim
        for hidden_dim in hidden_layers:
            self.branch_imag_layers.append(
                EnhancedMLPBlock(prev_dim, hidden_dim, cfg.deeponet)
            )
            prev_dim = hidden_dim

        # 虚部Branch输出层
        self.branch_imag_output = nn.Linear(prev_dim, output_dim)

        # 融合MLP - 替代原来的纯乘积融合
        # 实部分支融合
        self.fusion_real = FusionMLP(
            branch_dim=output_dim,
            trunk_dim=output_dim,
            hidden_dims=[output_dim*2, output_dim],  # 两层融合MLP
            output_dim=output_dim,
            activation=cfg.deeponet.activation
        )

        # 虚部分支融合
        self.fusion_imag = FusionMLP(
            branch_dim=output_dim,
            trunk_dim=output_dim,
            hidden_dims=[output_dim*2, output_dim],  # 两层融合MLP
            output_dim=output_dim,
            activation=cfg.deeponet.activation
        )

        # 最终输出层 - 分别输出实部和虚部
        self.output_real = nn.Linear(output_dim, 1)
        self.output_imag = nn.Linear(output_dim, 1)

        # 初始化探针校正模块
        self.probe_correction = create_probe_correction_module(cfg)

        # 网络权重初始化
        self._initialize_weights()

    def _initialize_weights(self):
        """初始化网络权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                if self.cfg.deeponet.initializer == "Glorot normal":
                    nn.init.xavier_normal_(module.weight)
                elif self.cfg.deeponet.initializer == "He normal":
                    nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
                else:
                    # 默认初始化
                    nn.init.normal_(module.weight, 0, 0.02)

                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, branch_real_input, branch_imag_input, trunk_input,
                mask=None, lengths=None, return_correction_info=False, probe_true_values=None):
        """
        增强版双分支前向传播，包含探针校正
        Args:
            branch_real_input: [batch_size, probe_count*5] 实部探针数据 (x,y,z,freq,real_measurement)
            branch_imag_input: [batch_size, probe_count*5] 虚部探针数据 (x,y,z,freq,imag_measurement)
            trunk_input: [batch_size, N, 3] 坐标数据
            mask: [batch_size, N] 探针位置mask (用于探针校正)
            lengths: [batch_size] 有效长度 (用于探针校正)
            return_correction_info: bool 是否返回校正信息
            probe_true_values: [batch_size, probe_count, 2] 真实探针值 (用于RBF校正)
        Returns:
            torch.Tensor: [batch_size, N, 2] (real, imag)
            (可选) Dict: 校正信息
        """
        # 处理输入维度
        original_branch_real_shape = branch_real_input.shape
        original_branch_imag_shape = branch_imag_input.shape
        original_trunk_shape = trunk_input.shape

        # 确保batch_size一致
        if branch_real_input.dim() == 1:
            branch_real_input = branch_real_input.unsqueeze(0)
            single_sample = True
        else:
            single_sample = False

        if branch_imag_input.dim() == 1:
            branch_imag_input = branch_imag_input.unsqueeze(0)
        elif single_sample and branch_imag_input.dim() == 2:
            branch_imag_input = branch_imag_input.unsqueeze(0)

        if trunk_input.dim() == 2:
            trunk_input = trunk_input.unsqueeze(0)
        elif trunk_input.dim() == 3 and single_sample:
            trunk_input = trunk_input.unsqueeze(0)

        batch_size = trunk_input.shape[0]
        num_points = trunk_input.shape[1]

        # 验证输入维度
        expected_branch_dim = self.cfg.deeponet.probe_count * 5

        if branch_real_input.shape[-1] != expected_branch_dim:
            raise ValueError(f"实部分支输入维度错误: 期望 {expected_branch_dim}, 实际 {branch_real_input.shape[-1]}")
        if branch_imag_input.shape[-1] != expected_branch_dim:
            raise ValueError(f"虚部分支输入维度错误: 期望 {expected_branch_dim}, 实际 {branch_imag_input.shape[-1]}")

        # Trunk网络前向传播 - 共享给两个分支
        trunk_flat = trunk_input.view(-1, 3)  # [batch_size * num_points, 3]
        trunk_out = trunk_flat

        for layer in self.trunk_layers:
            trunk_out = layer(trunk_out)
        trunk_out = self.trunk_output(trunk_out)
        trunk_out = trunk_out.view(batch_size, num_points, -1)  # [batch_size, N, output_dim]

        # 实部Branch网络前向传播
        branch_real_out = branch_real_input
        for layer in self.branch_real_layers:
            branch_real_out = layer(branch_real_out)
        branch_real_out = self.branch_real_output(branch_real_out)

        # 虚部Branch网络前向传播
        branch_imag_out = branch_imag_input
        for layer in self.branch_imag_layers:
            branch_imag_out = layer(branch_imag_out)
        branch_imag_out = self.branch_imag_output(branch_imag_out)

        # 扩展branch输出以匹配trunk的所有点
        branch_real_expanded = branch_real_out.unsqueeze(1).expand(-1, num_points, -1)  # [batch_size, N, output_dim]
        branch_imag_expanded = branch_imag_out.unsqueeze(1).expand(-1, num_points, -1)  # [batch_size, N, output_dim]

        # 加性融合MLP：替代原来的纯乘积融合
        combined_real = self.fusion_real(branch_real_expanded, trunk_out)  # [batch_size, N, output_dim]
        combined_imag = self.fusion_imag(branch_imag_expanded, trunk_out)  # [batch_size, N, output_dim]

        # 分别输出实部和虚部
        real_output = self.output_real(combined_real)  # [batch_size, N, 1]
        imag_output = self.output_imag(combined_imag)  # [batch_size, N, 1]

        # 合并为基线输出 [batch_size, N, 2] (real, imag)
        baseline_output = torch.cat([real_output, imag_output], dim=-1)

        # 应用探针校正 - 传递真实探针值
        final_output, correction_info = self.probe_correction(
            baseline_output, branch_real_input, trunk_input, mask, lengths,
            probe_true_values=probe_true_values
        )

        # 如果是单样本输入，去掉batch维度
        if single_sample:
            final_output = final_output.squeeze(0)  # [N, 2]

        if return_correction_info:
            return final_output, correction_info
        else:
            return final_output

    def get_model_info(self):
        """获取模型信息"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        info = {
            'model_type': 'EnhancedPyTorchDualBranchDeepONet',
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'network_preset': self.cfg.deeponet.network_preset,
            'hidden_layers': self.cfg.deeponet.hidden_layers,
            'output_dim': self.cfg.deeponet.output_dim,
            'activation': self.cfg.deeponet.activation,
            'dropout_rate': self.cfg.deeponet.dropout_rate,
            'use_residual': self.cfg.deeponet.use_residual,
            'use_attention': self.cfg.deeponet.use_attention,
            'probe_count': self.cfg.deeponet.probe_count,
            'input_dim': self.cfg.deeponet.input_dim
        }

        return info

    def print_model_info(self):
        """打印模型信息"""
        info = self.get_model_info()
        print("\n" + "="*60)
        print(f"模型类型: {info['model_type']}")
        print(f"网络预设: {info['network_preset']}")
        print(f"隐藏层结构: {info['hidden_layers']}")
        print(f"特征维度: {info['output_dim']}")
        print(f"激活函数: {info['activation']}")
        print(f"Dropout率: {info['dropout_rate']}")
        print(f"残差连接: {'启用' if info['use_residual'] else '禁用'}")
        print(f"注意力机制: {'启用' if info['use_attention'] else '禁用'}")
        print(f"探针数量: {info['probe_count']}")
        print(f"输入维度: {info['input_dim']}")
        print("-"*60)
        print(f"总参数数量: {info['total_parameters']:,}")
        print(f"可训练参数: {info['trainable_parameters']:,}")
        print("="*60)


def create_enhanced_deeponet(cfg: Config):
    """
    创建增强版单Branch DeepONet模型

    Args:
        cfg: 配置对象

    Returns:
        SingleBranchDeepONet: 单Branch DeepONet模型
    """
    model = SingleBranchDeepONet(cfg)
    model.print_model_info()
    return model


def count_parameters(model):
    """计算模型参数数量（兼容函数）"""
    info = model.get_model_info() if hasattr(model, 'get_model_info') else {}
    total_params = info.get('total_parameters', sum(p.numel() for p in model.parameters()))
    trainable_params = info.get('trainable_parameters',
                               sum(p.numel() for p in model.parameters() if p.requires_grad))

    print(f"总参数数量: {total_params:,}")
    print(f"可训练参数数量: {trainable_params:,}")

    return total_params, trainable_params


def test_enhanced_model():
    """测试增强版模型"""
    print("测试增强版DeepONet模型...")

    # 测试不同预设
    presets = ['lightweight', 'standard', 'heavy', 'ultra']

    for preset in presets:
        print(f"\n{'='*60}")
        print(f"测试预设: {preset}")
        print(f"{'='*60}")

        try:
            # 创建配置
            cfg = Config()
            cfg.deeponet.set_network_preset(preset)  # 使用新方法设置预设

            # 创建模型
            model = create_enhanced_deeponet(cfg)

            # 测试前向传播
            batch_size = 2
            num_probes = cfg.deeponet.probe_count
            num_points = 128

            branch_real_input = torch.randn(batch_size, num_probes * 5)
            branch_imag_input = torch.randn(batch_size, num_probes * 5)
            trunk_input = torch.randn(batch_size, num_points, 3)

            with torch.no_grad():
                output = model(branch_real_input, branch_imag_input, trunk_input)
                print(f"输出形状: {output.shape}")
                print(f"预设 {preset} 测试成功！")

        except Exception as e:
            print(f"预设 {preset} 测试失败: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    test_enhanced_model()