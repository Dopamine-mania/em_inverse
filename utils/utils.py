"""
工具函数模块
"""

import torch
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib
import json

# 设置matplotlib使用非交互式后端和中文字体支持
matplotlib.use('Agg')
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

import yaml

# 添加配置保存辅助函数
def save_config_to_yaml(cfg, yaml_path):
    """保存配置到YAML文件"""
    try:
        config_dict = {
            'data': {
                'data_path': cfg.data.data_path,
                'num_probes': cfg.data.num_probes,
                'max_samples': cfg.data.max_samples,
                'max_frequency': cfg.data.max_frequency,
                'normalize_data': cfg.data.normalize_data,
                'fixed_probe_positions': cfg.data.fixed_probe_positions
            },
            'deeponet': {
                'network_preset': cfg.deeponet.network_preset,
                'hidden_layers': cfg.deeponet.hidden_layers,
                'output_dim': cfg.deeponet.output_dim,
                'activation': cfg.deeponet.activation,
                'dropout_rate': cfg.deeponet.dropout_rate,
                'use_attention': cfg.deeponet.use_attention,
                'use_residual': cfg.deeponet.use_residual
            },
            'training': {
                'epochs': cfg.training.epochs,
                'batch_size': cfg.training.batch_size,
                'learning_rate': cfg.training.learning_rate,
                'display_every': cfg.training.display_every,
                'save_interval': cfg.training.save_interval,
                'plot_frequency': cfg.training.plot_frequency,
                'early_stopping_patience': cfg.training.early_stopping_patience,
                'min_delta': cfg.training.min_delta
            },
            'physics': {
                'probe_loss_weight': cfg.physics.probe_loss_weight,
                'field_loss_weight': cfg.physics.field_loss_weight,
                'correlation_loss_weight': cfg.physics.correlation_loss_weight,
                'smoothness_loss_weight': cfg.physics.smoothness_loss_weight,
                'enable_rbf_correction': cfg.physics.enable_rbf_correction,
                'rbf_gamma': cfg.physics.rbf_gamma,
                'learn_rbf_gamma': cfg.physics.learn_rbf_gamma,
                'rbf_frequency_aware': cfg.physics.rbf_frequency_aware,
                'probe_alignment_warning': cfg.physics.probe_alignment_warning
            }
        }

        with open(yaml_path, 'w', encoding='utf-8') as f:
            yaml.dump(config_dict, f, default_flow_style=False, allow_unicode=True, indent=2)

        print(f"[CONFIG SAVED] {yaml_path}")

    except Exception as e:
        print(f"[WARNING] Failed to save config to YAML: {e}")


from config import Config


def save_checkpoint(model, optimizer, epoch: int, loss_history: list, cfg: Config,
                   save_dir: str = "./checkpoints", model_type: str = "deeponet",
                   is_best_model: bool = False, is_final: bool = False):
    """
    增强版模型检查点保存

    Args:
        model: 模型实例
        optimizer: 优化器实例
        epoch: 当前epoch
        loss_history: 损失历史
        cfg: 配置对象
        save_dir: 保存目录
        model_type: 模型类型
        is_best_model: 是否是最佳模型
        is_final: 是否是最终模型
    """
    save_dir = Path(save_dir).resolve()  # 转换为绝对路径
    save_dir.mkdir(parents=True, exist_ok=True)  # 添加parents=True

    # 提取详细配置用于保存
    detailed_config = {
        # 数据配置
        'data.num_probes': cfg.data.num_probes,
        'data.max_samples': cfg.data.max_samples,
        'data.max_frequency': cfg.data.max_frequency,

        # DeepONet网络配置
        'deeponet.network_preset': cfg.deeponet.network_preset,
        'deeponet.hidden_layers': cfg.deeponet.hidden_layers,
        'deeponet.output_dim': cfg.deeponet.output_dim,
        'deeponet.activation': cfg.deeponet.activation,
        'deeponet.dropout_rate': cfg.deeponet.dropout_rate,
        'deeponet.use_attention': cfg.deeponet.use_attention,
        'deeponet.use_residual': cfg.deeponet.use_residual,

        # 训练配置
        'training.epochs': cfg.training.epochs,
        'training.batch_size': cfg.training.batch_size,
        'training.learning_rate': cfg.training.learning_rate,

        # 物理配置
        'physics.enable_rbf_correction': cfg.physics.enable_rbf_correction,
        'physics.probe_loss_weight': cfg.physics.probe_loss_weight,
        'physics.field_loss_weight': cfg.physics.field_loss_weight,
        'physics.correlation_loss_weight': cfg.physics.correlation_loss_weight,
        'physics.smoothness_loss_weight': cfg.physics.smoothness_loss_weight,
        'physics.rbf_gamma': cfg.physics.rbf_gamma,
        'physics.learn_rbf_gamma': cfg.physics.learn_rbf_gamma,
        'physics.rbf_frequency_aware': cfg.physics.rbf_frequency_aware,
    }

    if model_type in ["pytorch", "pytorch_optimized"]:
        # PyTorch模型保存
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss_history': loss_history,
            'config_params': detailed_config,  # 保存详细配置
            'model_type': model_type,
            'training_date': str(Path(__file__).parent),
            'architecture_summary': {
                'total_parameters': sum(p.numel() for p in model.parameters()),
                'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad)
            }
        }

        # 保存文件名策略
        if is_best_model:
            # 最佳模型 - 包含epoch和loss信息
            best_path = save_dir / f"best_epoch_{epoch:04d}_loss_{loss_history[-1] if loss_history else 0.0:.6f}.pth"
            # 同时保存一个通用的best文件
            best_generic_path = save_dir / "pytorch_deeponet_best.pth"

            torch.save(checkpoint, best_path)
            torch.save(checkpoint, best_generic_path)

            print(f"[BEST MODEL SAVED]")
            print(f"  Detailed: {best_path}")
            print(f"  Generic:  {best_generic_path}")
            print(f"  Epoch: {epoch}, Loss: {loss_history[-1] if loss_history else 0.0:.6f}")

        elif is_final:
            # 最终模型
            final_path = save_dir / f"final_epoch_{epoch:04d}_loss_{loss_history[-1] if loss_history else 0.0:.6f}.pth"
            torch.save(checkpoint, final_path)
            print(f"[FINAL MODEL SAVED] {final_path} (Epoch {epoch})")

        else:
            # 定期保存（如果需要的话）
            regular_path = save_dir / f"checkpoint_epoch_{epoch:04d}.pth"
            torch.save(checkpoint, regular_path)
            print(f"[CHECKPOINT SAVED] {regular_path} (Epoch {epoch})")

        # 同时保存配置到单独的YAML文件用于参考
        config_yaml_path = save_dir / "checkpoint_config.yaml"
        save_config_to_yaml(cfg, config_yaml_path)

    elif model_type == "deepxde":
        # DeepXDE模型保存
        try:
            model_path = save_dir / f"deepxde_deeponet_epoch_{epoch}.ckpt"
            model.save(str(model_path), protocol="pickle")
            print(f"DeepXDE检查点已保存: {model_path}")

            # 保存训练状态
            state_path = save_dir / f"deepxde_state_epoch_{epoch}.json"
            training_state = {
                'epoch': epoch,
                'loss_history': loss_history,
                'config': cfg.to_dict()
            }
            with open(state_path, 'w') as f:
                json.dump(training_state, f, indent=2)

        except Exception as e:
            print(f"DeepXDE检查点保存失败: {e}")


def load_checkpoint(checkpoint_path: str, model, optimizer=None, device=None):
    """
    加载模型检查点

    Args:
        checkpoint_path: 检查点路径
        model: 模型实例
        optimizer: 优化器实例
        device: 计算设备

    Returns:
        epoch, loss_history, config
    """
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"检查点文件不存在: {checkpoint_path}")

    if checkpoint_path.suffix == '.pth':
        # PyTorch检查点
        checkpoint = torch.load(checkpoint_path, map_location=device)

        model.load_state_dict(checkpoint['model_state_dict'])

        if optimizer and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        epoch = checkpoint.get('epoch', 0)
        loss_history = checkpoint.get('loss_history', [])
        config_dict = checkpoint.get('config', {})

        print(f"PyTorch检查点已加载: {checkpoint_path}")
        return epoch, loss_history, config_dict

    else:
        raise ValueError(f"不支持的检查点格式: {checkpoint_path.suffix}")


def visualize_predictions(y_true, y_pred, coords, mask=None, save_path=None, title=None):
    """
    可视化预测结果

    Args:
        y_true: [N, 2] 真实值
        y_pred: [N, 2] 预测值
        coords: [N, 3] 坐标 (x, y, freq)
        mask: [N] 探针位置掩码
        save_path: 保存路径
        title: 图片标题
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # 提取坐标
    x = coords[:, 0]
    y = coords[:, 1]

    # 真实值实部
    scatter1 = axes[0, 0].scatter(x, y, c=y_true[:, 0], cmap='viridis', s=10)
    axes[0, 0].set_title('真实值 - 实部')
    axes[0, 0].set_xlabel('X')
    axes[0, 0].set_ylabel('Y')
    plt.colorbar(scatter1, ax=axes[0, 0])

    # 真实值虚部
    scatter2 = axes[0, 1].scatter(x, y, c=y_true[:, 1], cmap='viridis', s=10)
    axes[0, 1].set_title('真实值 - 虚部')
    axes[0, 1].set_xlabel('X')
    axes[0, 1].set_ylabel('Y')
    plt.colorbar(scatter2, ax=axes[0, 1])

    # 真实值幅值
    magnitude_true = np.sqrt(y_true[:, 0]**2 + y_true[:, 1]**2)
    scatter3 = axes[0, 2].scatter(x, y, c=magnitude_true, cmap='viridis', s=10)
    axes[0, 2].set_title('真实值 - 幅值')
    axes[0, 2].set_xlabel('X')
    axes[0, 2].set_ylabel('Y')
    plt.colorbar(scatter3, ax=axes[0, 2])

    # 预测值实部
    scatter4 = axes[1, 0].scatter(x, y, c=y_pred[:, 0], cmap='viridis', s=10)
    axes[1, 0].set_title('预测值 - 实部')
    axes[1, 0].set_xlabel('X')
    axes[1, 0].set_ylabel('Y')
    plt.colorbar(scatter4, ax=axes[1, 0])

    # 预测值虚部
    scatter5 = axes[1, 1].scatter(x, y, c=y_pred[:, 1], cmap='viridis', s=10)
    axes[1, 1].set_title('预测值 - 虚部')
    axes[1, 1].set_xlabel('X')
    axes[1, 1].set_ylabel('Y')
    plt.colorbar(scatter5, ax=axes[1, 1])

    # 预测值幅值
    magnitude_pred = np.sqrt(y_pred[:, 0]**2 + y_pred[:, 1]**2)
    scatter6 = axes[1, 2].scatter(x, y, c=magnitude_pred, cmap='viridis', s=10)
    axes[1, 2].set_title('预测值 - 幅值')
    axes[1, 2].set_xlabel('X')
    axes[1, 2].set_ylabel('Y')
    plt.colorbar(scatter6, ax=axes[1, 2])

    # 标记探针位置
    if mask is not None:
        probe_x = x[mask]
        probe_y = y[mask]

        for ax in axes.flat:
            ax.scatter(probe_x, probe_y, c='red', s=20, marker='x', alpha=0.7)

    if title:
        fig.suptitle(title, fontsize=16)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"可视化结果已保存: {save_path}")

    plt.show()


def calculate_metrics(y_true, y_pred, mask=None):
    """
    计算评估指标

    Args:
        y_true: [N, 2] 真实值
        y_pred: [N, 2] 预测值
        mask: [N] 探针位置掩码

    Returns:
        评估指标字典
    """
    metrics = {}

    # 整体误差
    mse_all = np.mean((y_true - y_pred)**2)
    rmse_all = np.sqrt(mse_all)
    mae_all = np.mean(np.abs(y_true - y_pred))

    metrics['overall_mse'] = mse_all
    metrics['overall_rmse'] = rmse_all
    metrics['overall_mae'] = mae_all

    # 实部和虚部分别计算
    mse_real = np.mean((y_true[:, 0] - y_pred[:, 0])**2)
    mse_imag = np.mean((y_true[:, 1] - y_pred[:, 1])**2)

    metrics['real_mse'] = mse_real
    metrics['imag_mse'] = mse_imag

    # 幅值误差
    magnitude_true = np.sqrt(y_true[:, 0]**2 + y_true[:, 1]**2)
    magnitude_pred = np.sqrt(y_pred[:, 0]**2 + y_pred[:, 1]**2)

    mse_magnitude = np.mean((magnitude_true - magnitude_pred)**2)
    metrics['magnitude_mse'] = mse_magnitude

    # 探针位置误差
    if mask is not None:
        probe_true = y_true[mask]
        probe_pred = y_pred[mask]

        if len(probe_true) > 0:
            mse_probe = np.mean((probe_true - probe_pred)**2)
            rmse_probe = np.sqrt(mse_probe)
            mae_probe = np.mean(np.abs(probe_true - probe_pred))

            metrics['probe_mse'] = mse_probe
            metrics['probe_rmse'] = rmse_probe
            metrics['probe_mae'] = mae_probe

    # 相对误差
    if np.any(y_true != 0):
        relative_error = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8)))
        metrics['relative_error'] = relative_error

    return metrics


def print_metrics(metrics):
    """打印评估指标"""
    print("\n" + "="*50)
    print("评估指标")
    print("="*50)

    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key:20s}: {value:.6e}")
        else:
            print(f"{key:20s}: {value}")

    print("="*50)


def set_random_seed(seed: int = 42):
    """设置随机种子"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def get_model_summary(model, input_size=None):
    """获取模型摘要"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("\n模型摘要")
    print("-" * 40)
    print(f"总参数数量: {total_params:,}")
    print(f"可训练参数: {trainable_params:,}")
    print(f"非可训练参数: {total_params - trainable_params:,}")

    if input_size:
        # 计算模型大小（MB）
        param_size = 0
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()

        buffer_size = 0
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()

        model_size = (param_size + buffer_size) / 1024 / 1024
        print(f"模型大小: {model_size:.2f} MB")

    print("-" * 40)