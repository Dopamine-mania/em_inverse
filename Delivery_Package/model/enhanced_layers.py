"""
神经网路增强模块
包含残差连接、注意力机制、Dropout等功能
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ResidualBlock(nn.Module):
    """残差连接块"""

    def __init__(self, input_dim, output_dim, activation='relu', dropout_rate=0.0):
        super(ResidualBlock, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_projection = input_dim != output_dim

        # 主路径
        self.linear1 = nn.Linear(input_dim, output_dim)
        self.activation = self._get_activation(activation)
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()
        self.linear2 = nn.Linear(output_dim, output_dim)

        # 投影连接（如果维度不匹配）
        if self.use_projection:
            self.projection = nn.Linear(input_dim, output_dim)

        # 层归一化
        self.layer_norm = nn.LayerNorm(output_dim)

    def _get_activation(self, activation_name):
        """获取激活函数"""
        activation_name = activation_name.lower()
        if activation_name == "relu":
            return nn.ReLU()
        elif activation_name == "gelu":
            return nn.GELU()
        elif activation_name == "swish":
            return nn.SiLU()
        elif activation_name == "leaky_relu":
            return nn.LeakyReLU(0.2)
        elif activation_name == "tanh":
            return nn.Tanh()
        else:
            return nn.ReLU()

    def forward(self, x):
        residual = x

        # 主路径
        out = self.linear1(x)
        out = self.activation(out)
        out = self.dropout(out)
        out = self.linear2(out)

        # 投影连接（如果需要）
        if self.use_projection:
            residual = self.projection(residual)

        # 残差连接 + 层归一化
        out = out + residual
        out = self.layer_norm(out)

        return out


class MultiHeadAttention(nn.Module):
    """多头注意力机制"""

    def __init__(self, d_model, num_heads=8, dropout_rate=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        """缩放点积注意力"""
        d_k = Q.size(-1)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        return torch.matmul(attention_weights, V), attention_weights

    def forward(self, x):
        batch_size = x.size(0)

        # 线性变换并分割为多头
        Q = self.w_q(x).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(x).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(x).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        # 注意力计算
        attention_output, _ = self.scaled_dot_product_attention(Q, K, V)

        # 连接多头并输出
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model
        )

        return self.w_o(attention_output)


class AttentionBlock(nn.Module):
    """注意力块（包含前馈网络）"""

    def __init__(self, d_model, num_heads=8, dropout_rate=0.1, activation='gelu'):
        super(AttentionBlock, self).__init__()

        self.attention = MultiHeadAttention(d_model, num_heads, dropout_rate)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # 前馈网络
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            self._get_activation(activation),
            nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity(),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()
        )

    def _get_activation(self, activation_name):
        """获取激活函数"""
        activation_name = activation_name.lower()
        if activation_name == "relu":
            return nn.ReLU()
        elif activation_name == "gelu":
            return nn.GELU()
        elif activation_name == "swish":
            return nn.SiLU()
        elif activation_name == "leaky_relu":
            return nn.LeakyReLU(0.2)
        elif activation_name == "tanh":
            return nn.Tanh()
        else:
            return nn.ReLU()

    def forward(self, x):
        # 注意力子层 + 残差连接
        attn_output = self.attention(self.norm1(x))
        x = x + attn_output

        # 前馈网络子层 + 残差连接
        ffn_output = self.ffn(self.norm2(x))
        x = x + ffn_output

        return x


class EnhancedMLPBlock(nn.Module):
    """增强的MLP块，支持残差、dropout、注意力"""

    def __init__(self, input_dim, output_dim, cfg):
        super(EnhancedMLPBlock, self).__init__()

        self.cfg = cfg
        self.use_residual = cfg.use_residual
        self.use_attention = cfg.use_attention

        # 主线性层
        self.linear = nn.Linear(input_dim, output_dim)

        # Dropout
        self.dropout = nn.Dropout(cfg.dropout_rate) if cfg.dropout_rate > 0 else nn.Identity()

        # 激活函数
        self.activation = self._get_activation(cfg.activation)

        # 残差连接
        if self.use_residual:
            self.residual_block = ResidualBlock(
                input_dim, output_dim, cfg.activation, cfg.dropout_rate
            )

        # 注意力机制（只在较高维度使用）
        if self.use_attention and output_dim >= 128:
            self.attention_block = AttentionBlock(
                output_dim, num_heads=8, dropout_rate=cfg.dropout_rate, activation=cfg.activation
            )
            self.attention_norm = nn.LayerNorm(output_dim)

    def _get_activation(self, activation_name):
        """获取激活函数"""
        activation_name = activation_name.lower()
        if activation_name == "tanh":
            return nn.Tanh()
        elif activation_name == "relu":
            return nn.ReLU()
        elif activation_name == "gelu":
            return nn.GELU()
        elif activation_name == "swish":
            return nn.SiLU()
        elif activation_name == "leaky_relu":
            return nn.LeakyReLU(0.2)
        else:
            return nn.ReLU()

    def forward(self, x):
        if self.use_residual:
            # 使用残差连接
            out = self.residual_block(x)
        else:
            # 标准MLP
            out = self.linear(x)
            out = self.activation(out)
            out = self.dropout(out)

        # 注意力机制
        if self.use_attention and hasattr(self, 'attention_block'):
            # 如果需要，添加序列维度
            if out.dim() == 2:
                out = out.unsqueeze(1)
                out = self.attention_block(out)
                out = out.squeeze(1)
            else:
                out = self.attention_block(out)
            out = self.attention_norm(out)

        return out


class FusionMLP(nn.Module):
    """加性融合MLP：将branch和trunk特征concat后通过多层MLP交互融合"""

    def __init__(self, branch_dim, trunk_dim, hidden_dims, output_dim, activation='gelu'):
        super(FusionMLP, self).__init__()

        # 输入是branch和trunk的拼接
        input_dim = branch_dim + trunk_dim

        # 构建多层MLP
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                self._get_activation(activation),
                nn.LayerNorm(hidden_dim)  # 添加层归一化提升稳定性
            ])
            prev_dim = hidden_dim

        # 输出层
        layers.append(nn.Linear(prev_dim, output_dim))

        self.mlp = nn.Sequential(*layers)

    def _get_activation(self, activation_name):
        """获取激活函数"""
        activation_name = activation_name.lower()
        if activation_name == "relu":
            return nn.ReLU()
        elif activation_name == "gelu":
            return nn.GELU()
        elif activation_name == "swish":
            return nn.SiLU()
        elif activation_name == "leaky_relu":
            return nn.LeakyReLU(0.2)
        elif activation_name == "tanh":
            return nn.Tanh()
        else:
            return nn.GELU()

    def forward(self, branch_out, trunk_out):
        """
        Args:
            branch_out: [batch_size, N, branch_dim] 或 [batch_size, branch_dim]
            trunk_out: [batch_size, N, trunk_dim] 或 [batch_size, N, trunk_dim]
        Returns:
            [batch_size, N, output_dim] 或 [batch_size, N, output_dim]
        """
        # 确保维度匹配
        if branch_out.dim() == 2 and trunk_out.dim() == 3:
            # branch是全局特征，trunk是空间特征
            batch_size, num_points, trunk_dim = trunk_out.shape
            branch_out = branch_out.unsqueeze(1).expand(-1, num_points, -1)

        # 拼接特征
        fused = torch.cat([branch_out, trunk_out], dim=-1)  # [batch_size, N, branch_dim+trunk_dim]

        # 通过MLP交互融合
        output = self.mlp(fused)

        return output