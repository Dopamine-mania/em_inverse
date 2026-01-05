"""
DeepONet模型定义 - 清理版
用于电磁场重构的双分支DeepONet架构

当前支持的模型：
- PyTorchDualBranchDeepONet: 原始双分支架构（保持向后兼容）
- EnhancedPyTorchDualBranchDeepONet: 增强版双分支架构（推荐使用）

推荐使用：
    from model.enhanced_deeponet import create_enhanced_deeponet
    model = create_enhanced_deeponet(cfg)
"""

import numpy as np
import torch
import torch.nn as nn
import sys
from pathlib import Path

# 添加父目录到路径
sys.path.append(str(Path(__file__).parent.parent))

from config.config import Config


class PyTorchDualBranchDeepONet(nn.Module):
    """PyTorch实现的双分支DeepONet - 实部和虚部分别预测（原始版本，保持兼容性）"""

    def __init__(self, cfg: Config):
        super(PyTorchDualBranchDeepONet, self).__init__()
        self.cfg = cfg

        print("警告：这是原始版本的DeepONet，建议使用增强版本：")
        print("  from model.enhanced_deeponet import create_enhanced_deeponet")
        print("  model = create_enhanced_deeponet(cfg)")
        print()

        # Trunk网络 - 处理坐标，两个分支共享
        trunk_layers = [3] + cfg.deeponet.hidden_layers + [cfg.deeponet.output_dim]
        self.trunk_net = nn.ModuleList()

        for i in range(len(trunk_layers) - 1):
            self.trunk_net.append(nn.Linear(trunk_layers[i], trunk_layers[i+1]))
            if i < len(trunk_layers) - 2:  # 不在最后一层加激活函数
                self.trunk_net.append(self._get_activation(cfg.deeponet.activation))

        # Branch网络 - 分别处理实部和虚部探针数据
        branch_input_dim = cfg.deeponet.probe_count * 5

        # 实部Branch网络
        real_branch_layers = [branch_input_dim] + cfg.deeponet.hidden_layers + [cfg.deeponet.output_dim]
        self.branch_real_net = nn.ModuleList()
        for i in range(len(real_branch_layers) - 1):
            self.branch_real_net.append(nn.Linear(real_branch_layers[i], real_branch_layers[i+1]))
            if i < len(real_branch_layers) - 2:
                self.branch_real_net.append(self._get_activation(cfg.deeponet.activation))

        # 虚部Branch网络
        imag_branch_layers = [branch_input_dim] + cfg.deeponet.hidden_layers + [cfg.deeponet.output_dim]
        self.branch_imag_net = nn.ModuleList()
        for i in range(len(imag_branch_layers) - 1):
            self.branch_imag_net.append(nn.Linear(imag_branch_layers[i], imag_branch_layers[i+1]))
            if i < len(imag_branch_layers) - 2:
                self.branch_imag_net.append(self._get_activation(cfg.deeponet.activation))

        # 输出层 - 分别输出实部和虚部
        self.output_real = nn.Linear(cfg.deeponet.output_dim, 1)
        self.output_imag = nn.Linear(cfg.deeponet.output_dim, 1)

    def _get_activation(self, activation_name):
        """获取激活函数 - 支持更多类型"""
        activation_name = activation_name.lower()
        if activation_name == "tanh":
            return nn.Tanh()
        elif activation_name == "relu":
            return nn.ReLU()
        elif activation_name == "gelu":
            return nn.GELU()
        elif activation_name == "swish":
            return nn.SiLU()  # Swish激活函数
        elif activation_name == "leaky_relu":
            return nn.LeakyReLU(0.2)
        else:
            print(f"警告：未知激活函数 '{activation_name}'，使用ReLU")
            return nn.ReLU()

    def forward(self, branch_real_input, branch_imag_input, trunk_input):
        """
        双分支前向传播
        Args:
            branch_real_input: [batch_size, probe_count*5] 实部探针数据 (x,y,z,freq,real_measurement)
            branch_imag_input: [batch_size, probe_count*5] 虚部探针数据 (x,y,z,freq,imag_measurement)
            trunk_input: [batch_size, N, 3] 坐标数据
        Returns:
            torch.Tensor: [batch_size, N, 2] (real, imag)
        """
        # 处理输入维度
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
        if branch_real_input.shape[-1] != expected_branch_dim or branch_imag_input.shape[-1] != expected_branch_dim:
            raise ValueError(f"分支输入维度错误: 期望 {expected_branch_dim}, 实际 {branch_real_input.shape[-1]}, {branch_imag_input.shape[-1]}")

        # Trunk网络前向传播 - 共享给两个分支
        trunk_flat = trunk_input.view(-1, 3)  # [batch_size * num_points, 3]
        trunk_out_flat = trunk_flat
        for layer in self.trunk_net:
            trunk_out_flat = layer(trunk_out_flat)
        trunk_out = trunk_out_flat.view(batch_size, num_points, -1)  # [batch_size, N, output_dim]

        # 实部Branch网络前向传播
        branch_real_out = branch_real_input
        for layer in self.branch_real_net:
            branch_real_out = layer(branch_real_out)

        # 虚部Branch网络前向传播
        branch_imag_out = branch_imag_input
        for layer in self.branch_imag_net:
            branch_imag_out = layer(branch_imag_out)

        # 扩展branch输出以匹配trunk的所有点
        branch_real_expanded = branch_real_out.unsqueeze(1).expand(-1, num_points, -1)
        branch_imag_expanded = branch_imag_out.unsqueeze(1).expand(-1, num_points, -1)

        # DeepONet组合: 逐元素乘积
        combined_real = branch_real_expanded * trunk_out
        combined_imag = branch_imag_expanded * trunk_out

        # 分别输出实部和虚部
        real_output = self.output_real(combined_real)
        imag_output = self.output_imag(combined_imag)

        # 合并为最终输出 [batch_size, N, 2] (real, imag)
        final_output = torch.cat([real_output, imag_output], dim=-1)

        # 如果是单样本输入，去掉batch维度
        if single_sample:
            final_output = final_output.squeeze(0)  # [N, 2]

        return final_output


def create_pytorch_dual_branch_deeponet(cfg: Config):
    """创建PyTorch版本的双分支DeepONet模型（原始版本，保持兼容性）"""
    return PyTorchDualBranchDeepONet(cfg)


def count_parameters(model):
    """计算模型参数数量"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"总参数数量: {total_params:,}")
    print(f"可训练参数数量: {trainable_params:,}")

    return total_params, trainable_params


def initialize_weights(model, initializer_type="Glorot normal"):
    """权重初始化"""
    if initializer_type == "Glorot normal":
        for m in model.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    elif initializer_type == "He normal":
        for m in model.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)


if __name__ == "__main__":
    # 测试原始模型
    cfg = Config()
    model = create_pytorch_dual_branch_deeponet(cfg)
    count_parameters(model)