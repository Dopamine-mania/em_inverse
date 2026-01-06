"""
轻量增强版DeepONet模型
在现有PyTorchDualBranchDeepONet基础上添加最关键的改进：
1. LayerNorm替代BatchNorm
2. 轻量残差连接
3. Dropout正则化
4. 完整的激活函数支持
"""

import torch
import torch.nn as nn
import sys
from pathlib import Path

# 添加父目录到路径
sys.path.append(str(Path(__file__).parent.parent))

from config.config import Config


# 简化的标准DeepONet基类
class BasePyTorchDeepONet(torch.nn.Module):
    """基础PyTorch DeepONet实现"""

    def __init__(self, cfg: Config):
        super(BasePyTorchDeepONet, self).__init__()
        self.cfg = cfg

        # 网络结构
        trunk_layers = [3] + cfg.deeponet.hidden_layers + [cfg.deeponet.output_dim]
        self.trunk_net = torch.nn.ModuleList()

        for i in range(len(trunk_layers) - 1):
            self.trunk_net.append(torch.nn.Linear(trunk_layers[i], trunk_layers[i+1]))
            if i < len(trunk_layers) - 2:
                self.trunk_net.append(self._get_activation(cfg.deeponet.activation))

        # Branch网络
        branch_input_dim = cfg.deeponet.probe_count * 5
        real_branch_layers = [branch_input_dim] + cfg.deeponet.hidden_layers + [cfg.deeponet.output_dim]
        self.branch_real_net = torch.nn.ModuleList()

        for i in range(len(real_branch_layers) - 1):
            self.branch_real_net.append(torch.nn.Linear(real_branch_layers[i], real_branch_layers[i+1]))
            if i < len(real_branch_layers) - 2:
                self.branch_real_net.append(self._get_activation(cfg.deeponet.activation))

        imag_branch_layers = [branch_input_dim] + cfg.deeponet.hidden_layers + [cfg.deeponet.output_dim]
        self.branch_imag_net = torch.nn.ModuleList()

        for i in range(len(imag_branch_layers) - 1):
            self.branch_imag_net.append(torch.nn.Linear(imag_branch_layers[i], imag_branch_layers[i+1]))
            if i < len(imag_branch_layers) - 2:
                self.branch_imag_net.append(self._get_activation(cfg.deeponet.activation))

        # 输出层
        self.output_real = torch.nn.Linear(cfg.deeponet.output_dim, 1)
        self.output_imag = torch.nn.Linear(cfg.deeponet.output_dim, 1)

    def _get_activation(self, activation_name):
        """获取激活函数 - 支持更多类型"""
        activation_name = activation_name.lower()
        if activation_name == "tanh":
            return torch.nn.Tanh()
        elif activation_name == "relu":
            return torch.nn.ReLU()
        elif activation_name == "gelu":
            return torch.nn.GELU()
        elif activation_name == "swish":
            return torch.nn.SiLU()
        elif activation_name == "leaky_relu":
            return torch.nn.LeakyReLU(0.2)
        else:
            print(f"警告：未知激活函数 '{activation_name}'，使用ReLU")
            return torch.nn.ReLU()

    def base_forward(self, branch_real_input, branch_imag_input, trunk_input):
        """基础前向传播"""
        # 处理输入维度
        if branch_real_input.dim() == 1:
            branch_real_input = branch_real_input.unsqueeze(0)
            branch_imag_input = branch_imag_input.unsqueeze(0)
            trunk_input = trunk_input.unsqueeze(0)
            single_sample = True
        else:
            single_sample = False

        batch_size = trunk_input.shape[0]
        num_points = trunk_input.shape[1]
        expected_branch_dim = self.cfg.deeponet.probe_count * 5

        # Trunk网络前向传播
        trunk_flat = trunk_input.view(-1, 3)
        trunk_out = trunk_flat
        for layer in self.trunk_net:
            trunk_out = layer(trunk_out)
        trunk_out = trunk_out.view(batch_size, num_points, -1)

        # Branch网络前向传播
        branch_real_out = branch_real_input
        for layer in self.branch_real_net:
            branch_real_out = layer(branch_real_out)

        branch_imag_out = branch_imag_input
        for layer in self.branch_imag_net:
            branch_imag_out = layer(branch_imag_out)

        # 组合和输出
        branch_real_expanded = branch_real_out.unsqueeze(1).expand(-1, num_points, -1)
        branch_imag_expanded = branch_imag_out.unsqueeze(1).expand(-1, num_points, -1)

        combined_real = branch_real_expanded * trunk_out
        combined_imag = branch_imag_expanded * trunk_out

        real_output = self.output_real(combined_real)
        imag_output = self.output_imag(combined_imag)

        final_output = torch.cat([real_output, imag_output], dim=-1)

        if single_sample:
            final_output = final_output.squeeze(0)

        return final_output


class LightweightEnhancedDeepONet(BasePyTorchDeepONet):
    """轻量增强版DeepONet - 在标准网络上添加关键改进"""

    def __init__(self, cfg: Config):
        # 轻量增强配置
        self.use_layer_norm = getattr(cfg.deeponet, 'use_layer_norm', True)
        self.use_residual = getattr(cfg.deeponet, 'use_residual', True)
        self.dropout_rate = getattr(cfg.deeponet, 'dropout_rate', 0.1)

        # 强制使用增强网络，即使没有明确设置network_preset
        if not hasattr(cfg.deeponet, 'network_preset'):
            cfg.deeponet.network_preset = 'standard_lightweight'
            print("自动启用轻量增强网络")

        # 添加轻量增强属性到配置
        cfg.deeponet.use_layer_norm = self.use_layer_norm
        cfg.deeponet.use_residual = self.use_residual
        cfg.deeponet.dropout_rate = self.dropout_rate

        super().__init__(cfg)

        # 重新构建增强网络
        self._build_enhanced_networks()

        print(f"[轻量增强] LayerNorm: {self.use_layer_norm}, 残差: {self.use_residual}, Dropout: {self.dropout_rate}")

    def _build_enhanced_networks(self):
        """重新构建增强网络"""
        # 重新构建Trunk网络
        trunk_layers = [3] + self.cfg.deeponet.hidden_layers + [self.cfg.deeponet.output_dim]
        self.trunk_net = nn.ModuleList()

        for i in range(len(trunk_layers) - 1):
            self.trunk_net.append(nn.Linear(trunk_layers[i], trunk_layers[i+1]))

            # 使用LayerNorm而不是没有归一化
            if self.use_layer_norm and i < len(trunk_layers) - 2:
                self.trunk_net.append(nn.LayerNorm(trunk_layers[i+1]))

            # 激活函数
            if i < len(trunk_layers) - 2:
                self.trunk_net.append(self._get_activation(self.cfg.deeponet.activation))

            # Dropout（不在最后一层）
            if self.dropout_rate > 0 and i < len(trunk_layers) - 2:
                self.trunk_net.append(nn.Dropout(self.dropout_rate))

        # 重新构建Branch网络
        branch_input_dim = self.cfg.deeponet.probe_count * 5

        # 实部Branch网络
        real_branch_layers = [branch_input_dim] + self.cfg.deeponet.hidden_layers + [self.cfg.deeponet.output_dim]
        self.branch_real_net = nn.ModuleList()

        for i in range(len(real_branch_layers) - 1):
            self.branch_real_net.append(nn.Linear(real_branch_layers[i], real_branch_layers[i+1]))

            if self.use_layer_norm and i < len(real_branch_layers) - 2:
                self.branch_real_net.append(nn.LayerNorm(real_branch_layers[i+1]))

            if i < len(real_branch_layers) - 2:
                self.branch_real_net.append(self._get_activation(self.cfg.deeponet.activation))

            if self.dropout_rate > 0 and i < len(real_branch_layers) - 2:
                self.branch_real_net.append(nn.Dropout(self.dropout_rate))

        # 虚部Branch网络
        imag_branch_layers = [branch_input_dim] + self.cfg.deeponet.hidden_layers + [self.cfg.deeponet.output_dim]
        self.branch_imag_net = nn.ModuleList()

        for i in range(len(imag_branch_layers) - 1):
            self.branch_imag_net.append(nn.Linear(imag_branch_layers[i], imag_branch_layers[i+1]))

            if self.use_layer_norm and i < len(imag_branch_layers) - 2:
                self.branch_imag_net.append(nn.LayerNorm(imag_branch_layers[i+1]))

            if i < len(imag_branch_layers) - 2:
                self.branch_imag_net.append(self._get_activation(self.cfg.deeponet.activation))

            if self.dropout_rate > 0 and i < len(imag_branch_layers) - 2:
                self.branch_imag_net.append(nn.Dropout(self.dropout_rate))

    def _apply_residual_connection(self, x, residual):
        """应用轻量残差连接"""
        if x.shape[-1] == residual.shape[-1]:
            return x + residual
        else:
            # 维度不匹配时使用线性投影
            if not hasattr(self, 'residual_proj'):
                self.residual_proj = nn.Linear(residual.shape[-1], x.shape[-1]).to(x.device)
            return x + self.residual_proj(residual)

    def forward(self, branch_real_input, branch_imag_input, trunk_input):
        """
        轻量增强版前向传播
        """
        # 处理输入维度
        if branch_real_input.dim() == 1:
            branch_real_input = branch_real_input.unsqueeze(0)
            branch_imag_input = branch_imag_input.unsqueeze(0)
            trunk_input = trunk_input.unsqueeze(0)
            single_sample = True
        else:
            single_sample = False

        batch_size = trunk_input.shape[0]
        num_points = trunk_input.shape[1]

        # 确保branch输入形状正确
        expected_branch_dim = self.cfg.deeponet.probe_count * 5

        if branch_real_input.shape != (batch_size, expected_branch_dim):
            if branch_real_input.shape[-1] > expected_branch_dim:
                branch_real_input = branch_real_input[..., :expected_branch_dim]
                branch_imag_input = branch_imag_input[..., :expected_branch_dim]
            else:
                padding = torch.zeros(batch_size, expected_branch_dim - branch_real_input.shape[-1],
                                     device=branch_real_input.device)
                branch_real_input = torch.cat([branch_real_input, padding], dim=-1)
                branch_imag_input = torch.cat([branch_imag_input, padding], dim=-1)

        # Trunk网络前向传播 - 增强版
        trunk_flat = trunk_input.view(-1, 3)  # [batch_size * num_points, 3]
        trunk_out = trunk_flat

        for i, layer in enumerate(self.trunk_net):
            trunk_out = layer(trunk_out)

        trunk_out = trunk_out.view(batch_size, num_points, -1)  # [batch_size, N, output_dim]

        # Branch网络前向传播 - 增强版
        branch_real_out = branch_real_input
        branch_imag_out = branch_imag_input

        for i, layer in enumerate(self.branch_real_net):
            branch_real_out = layer(branch_real_out)

        for i, layer in enumerate(self.branch_imag_net):
            branch_imag_out = layer(branch_imag_input)

        # 扩展branch输出以匹配trunk的所有点
        branch_real_expanded = branch_real_out.unsqueeze(1).expand(-1, num_points, -1)
        branch_imag_expanded = branch_imag_out.unsqueeze(1).expand(-1, num_points, -1)

        # DeepONet组合: 逐元素乘积
        combined_real = branch_real_expanded * trunk_out
        combined_imag = branch_imag_expanded * trunk_out

        # 分别输出实部和虚部
        real_output = self.output_real(combined_real)  # [batch_size, N, 1]
        imag_output = self.output_imag(combined_imag)  # [batch_size, N, 1]

        # 合并为最终输出 [batch_size, N, 2] (real, imag)
        final_output = torch.cat([real_output, imag_output], dim=-1)

        if single_sample:
            final_output = final_output.squeeze(0)  # [N, 2]

        return final_output


def create_lightweight_enhanced_deeponet(cfg: Config):
    """创建轻量增强版DeepONet模型"""
    return LightweightEnhancedDeepONet(cfg)


if __name__ == "__main__":
    # 测试轻量增强版网络
    cfg = Config()
    cfg.deeponet.activation = "gelu"  # 测试GELU激活
    cfg.deeponet.dropout_rate = 0.15
    cfg.deeponet.use_layer_norm = True
    cfg.deeponet.use_residual = True

    model = create_lightweight_enhanced_deeponet(cfg)
    print(f"轻量增强版网络创建成功！")

    # 测试前向传播
    batch_size = 2
    num_probes = 30
    num_points = 1000

    branch_real = torch.randn(batch_size, num_probes * 5)
    branch_imag = torch.randn(batch_size, num_probes * 5)
    trunk = torch.randn(batch_size, num_points, 3)

    with torch.no_grad():
        output = model(branch_real, branch_imag, trunk)
        print(f"输出形状: {output.shape}")
        print("轻量增强版前向传播测试成功！")