"""
电磁场重构DeepONet配置模块
"""
from dataclasses import dataclass
from typing import List, Tuple, Optional
import torch
import yaml
import os
from pathlib import Path


@dataclass
class PathsConfig:
    """路径相关配置 - 便于在不同环境中部署"""
    # 数据路径（更新为Linux路径 - 600个CSV文件）
    data_path: str = "/home/jovyan/teaching_material/Work/December/ai_physics/demands/01_电磁场数据文件"

    # 输出路径
    output_dir: str = "outputs"
    checkpoint_dir: str = "checkpoints"
    tensorboard_dir: str = "outputs/tensorboard"

    # 备用服务器路径 (注释掉，需要时取消注释)
    # server_data_path: str = "/home/user/data/电磁场数据文件"
    # server_output_dir: str = "/home/user/outputs"


@dataclass
class DataConfig:
    """数据相关配置"""
    # 数据路径 - 将从PathsConfig自动填充
    data_path: str = None  # 将在__post_init__中从paths填充

    # 网格配置
    grid_resolution: int = 128
    grid_bounds: Tuple[float, float, float, float] = (0.0, 4.5, 0.5, 5.5)

    # 探针配置
    num_probes: int = 25  # 固定探针数量（从50改为25）
    probe_sampling: str = "simple_grid"  # 使用简单网格采样
    fixed_probe_positions: bool = True  # 是否固定探针选取位置（改为True确保一致性）

    # 频率配置
    max_frequency: float = 20.0  # 频率限制(GHz)
    max_samples: int = None  # 最大样本数 (None表示使用所有样本)
    enable_frequency_scaling: bool = True  # 是否启用频率缩放（除以1GHz基准）
    frequency_scale_factor: float = 1000.0  # 频率缩放因子（MHz），1GHz = 1000MHz

    # 数据处理
    normalize_data: bool = False  # 禁用归一化操作
    use_complex_field: bool = True

    # 检查点
    checkpoint_dir: str = None  # 将在__post_init__中从paths填充

    def __post_init__(self):
        # 从paths配置填充路径，如果paths可用
        if hasattr(self, '_paths_config') and self._paths_config:
            if not self.data_path:
                self.data_path = self._paths_config.data_path
            if not self.checkpoint_dir:
                self.checkpoint_dir = self._paths_config.checkpoint_dir


@dataclass
class DeepONetConfig:
    """DeepONet特定配置"""
    # 网络结构 - input_dim 将根据探针数量自动计算
    input_dim: int = None  # branch输入维度，将在__post_init__中自动计算
    output_dim: int = 64  # 特征维度
    hidden_layers: List[int] = None
    activation: str = "relu"             # 改为ReLU，避免tanh的饱和问题
    initializer: str = "Glorot normal"

    # 增强网络配置
    dropout_rate: float = 0.1           # Dropout率，防止过拟合
    use_attention: bool = False         # 是否使用注意力机制
    use_residual: bool = True           # 是否使用残差连接
    network_preset: str = "lightweight" # 网络预设: lightweight, standard, heavy, ultra

    # 训练参数
    learning_rate: float = 1e-2          # 提高学习率，加快收敛
    epochs: int = 1000  # 修改为 epochs，保持一致性
    probe_count: int = 200  # 探针数量，将自动从data.num_probes获取

    def __post_init__(self):
        # 先设置默认的hidden_layers
        if self.hidden_layers is None:
            self.hidden_layers = [64, 64]  # 使用简单的模型结构，减少计算开销

        # 应用网络预设配置（会覆盖上面的默认值）
        self._apply_network_preset()

        # 自动计算input_dim = 探针数量 * 4个特征 (x, y, real, imag) + 1 (频率)
        # 单Branch架构：[x1,y1,r1,i1, x2,y2,r2,i2, ..., freq]
        if self.input_dim is None:
            self.input_dim = self.probe_count * 4 + 1

    def _apply_network_preset(self):
        """应用网络预设配置"""
        presets = {
            'lightweight': {
                'hidden_layers': [256, 256],
                'output_dim': 256,  # 增大表达维度
                'activation': 'gelu',
                'dropout_rate': 0.05,  # 轻量dropout
                'use_attention': False,  # 简化结构，轻量训练
                'use_residual': True  # 保留残差连接提升训练稳定性
            },
            'standard': {
                'hidden_layers': [256, 256, 256],
                'output_dim': 256,  # 提升到256维
                'activation': 'gelu',
                'dropout_rate': 0.1,  # 标准dropout
                'use_attention': True,  # 启用注意力机制
                'use_residual': True
            },
            'heavy': {
                'hidden_layers': [512, 512, 512, 256],
                'output_dim': 512,  # 提升到512维
                'activation': 'swish',
                'dropout_rate': 0.15,  # 强dropout
                'use_attention': True,  # 启用注意力机制
                'use_residual': True
            },
            'ultra': {
                'hidden_layers': [1024, 1024, 512, 512, 256],
                'output_dim': 512,  # 保持512维，避免参数爆炸
                'activation': 'gelu',
                'dropout_rate': 0.2,  # 高dropout率，防过拟合
                'use_attention': True,  # 启用注意力机制
                'use_residual': True
            }
        }

        if self.network_preset in presets:
            preset = presets[self.network_preset]
            # 应用预设值（只覆盖已存在的属性）
            for key, value in preset.items():
                if hasattr(self, key):
                    setattr(self, key, value)

    def set_network_preset(self, preset_name):
        """动态设置网络预设"""
        self.network_preset = preset_name
        self._apply_network_preset()

        # 同步input_dim
        if self.input_dim is None:
            self.input_dim = self.probe_count * 5


@dataclass
class TrainingConfig:
    """训练相关配置"""
    batch_size: int = 32
    epochs: int = 1000  # 修改为 epochs，与配置文件保持一致
    learning_rate: float = 1e-3

    # 训练设置
    train_test_split: float = 0.8  # 训练集比例 - 80%训练，20%测试，更合理的2:8比例
    save_interval: int = 50       # 保存间隔
    display_every: int = 10       # 显示间隔

    # 可视化设置
    plot_frequency: int = 2      # 默认每2个epoch保存一次可视化图片

    # 优化器配置
    optimizer: str = "adam"
    scheduler: str = "cosine"

    # 早停
    early_stopping_patience: int = 200     # 大幅增加耐心值，允许更多训练轮数
    min_delta: float = 1e-4             # 放宽最小改善阈值，避免过早停止


@dataclass
class PhysicsConfig:
    """物理损失相关配置"""
    # 损失权重 - 重新平衡，优先保证探针拟合
    probe_loss_weight: float = 50.0        # 探针对齐损失权重 - 提高优先级
    field_loss_weight: float = 10.0       # 场强差值损失权重 - 大幅降低
    correlation_loss_weight: float = 0.01  # 实部相关性损失权重 - 降低权重防止平凡场
    smoothness_loss_weight: float = 0   # 平滑正则损失权重
    spectral_loss_weight: float = 1.0   # Spectral Loss权重（k空间FFT损失）- Day 1核心

    # 探针损失参数 (DeepONet中不需要KNN)
    use_probe_weight: bool = False
    probe_alpha: float = 1.0

    # 平滑损失参数
    smooth_type: str = "laplacian"
    smooth_tau: float = 0.1

    # RBF插值校正参数
    enable_rbf_correction: bool = True  # 是否启用RBF探针校正
    rbf_gamma: float = 2.0            # RBF带宽参数 γ=1/σ²，控制插值平滑度
    learn_rbf_gamma: bool = False     # 是否学习RBF带宽参数
    rbf_frequency_aware: bool = False # 是否考虑频率差异（按频率分组插值）
    probe_alignment_warning: bool = True  # 是否启用探针对齐警告


@dataclass
class VisualizationConfig:
    """可视化相关配置"""
    output_dir: str = None  # 将在__post_init__中从paths填充
    save_images: bool = True
    plot_frequency: int = 10
    dpi: int = 150
    figsize: Tuple[int, int] = (15, 10)
    colormap: str = "viridis"

    def __post_init__(self):
        # 从paths配置填充路径，如果paths可用
        if hasattr(self, '_paths_config') and self._paths_config:
            if not self.output_dir:
                self.output_dir = self._paths_config.output_dir


@dataclass
class DeviceConfig:
    """设备和计算相关配置"""
    device: str = "auto"
    num_workers: int = 4
    pin_memory: bool = True
    mixed_precision: bool = False


class Config:
    """主配置类 - 支持YAML配置文件"""

    def __init__(self, config_file: Optional[str] = None, preset_name: Optional[str] = None):
        # 首先初始化默认配置
        self.paths = PathsConfig()

        # 初始化其他配置，并传递paths配置
        self.data = DataConfig()
        self.data._paths_config = self.paths
        # 触发DataConfig的__post_init__来填充路径
        if hasattr(self.data, '__post_init__'):
            self.data.__post_init__()

        self.deeponet = DeepONetConfig()
        self.training = TrainingConfig()
        self.physics = PhysicsConfig()
        self.visualization = VisualizationConfig()
        self.visualization._paths_config = self.paths
        # 触发VisualizationConfig的__post_init__来填充路径
        if hasattr(self.visualization, '__post_init__'):
            self.visualization.__post_init__()

        self.device = DeviceConfig()

        # 同步探针数量配置：确保一致性
        self.sync_probe_count()

        # 如果指定了配置文件，加载YAML配置
        if config_file:
            self.load_from_yaml(config_file, preset_name)

    def load_from_yaml(self, yaml_file: str, preset_name: Optional[str] = None):
        """从YAML文件加载配置"""
        yaml_path = Path(yaml_file)
        if not yaml_path.exists():
            print(f"警告：配置文件不存在: {yaml_path}")
            return

        try:
            with open(yaml_path, 'r', encoding='utf-8') as f:
                config_data = yaml.safe_load(f)

            # 直接从YAML文件加载各部分配置
            if 'paths' in config_data:
                self._apply_config_to_subconfig(self.paths, config_data['paths'])
                # 重新设置路径配置到各个子配置
                self.data.data_path = self.paths.data_path
                self.data.checkpoint_dir = self.paths.checkpoint_dir
                self.visualization.output_dir = self.paths.output_dir

            if 'data' in config_data:
                self._apply_config_to_subconfig(self.data, config_data['data'])

            if 'training' in config_data:
                self._apply_config_to_subconfig(self.training, config_data['training'])

            if 'output' in config_data:
                output_config = config_data['output']
                if 'output_dir' in output_config:
                    self.visualization.output_dir = output_config['output_dir']
                if 'checkpoint_dir' in output_config:
                    self.data.checkpoint_dir = output_config['checkpoint_dir']

            if 'physics' in config_data:
                self._apply_config_to_subconfig(self.physics, config_data['physics'])

            if 'device' in config_data:
                self._apply_config_to_subconfig(self.device, config_data['device'])

            # 加载deeponet配置
            if 'deeponet' in config_data:
                self._apply_config_to_subconfig(self.deeponet, config_data['deeponet'])
                # 重新应用网络预设，确保network_preset生效
                if hasattr(self.deeponet, 'network_preset'):
                    self.deeponet._apply_network_preset()

            # 同步探针数量: 从data.num_probes同步到deeponet.probe_count和input_dim
            if hasattr(self.data, 'num_probes'):
                self.sync_probe_count()

            print(f"成功加载YAML配置: {yaml_path}")

        except Exception as e:
            print(f"加载YAML配置失败: {e}")

    def get_available_presets(self, yaml_file: str = "config.yaml") -> dict:
        """获取YAML文件中可用的预设配置（简化版）"""
        return {"config": "使用config.yaml中的配置"}

    def _apply_config(self, config_dict: dict):
        """应用配置字典到相应的配置对象"""
        for key, value in config_dict.items():
            self.update(**{key: value})

    def _apply_config_to_subconfig(self, subconfig, config_dict: dict):
        """应用配置字典到子配置对象"""
        for key, value in config_dict.items():
            if hasattr(subconfig, key):
                setattr(subconfig, key, value)

    def get_available_presets(self, yaml_file: str = "config.yaml") -> dict:
        """获取YAML文件中可用的预设配置（简化版）"""
        return {"config": "使用config.yaml中的配置"}

    def get_device(self) -> torch.device:
        """获取计算设备"""
        if self.device.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.device.device)

    def update(self, **kwargs):
        """更新配置参数"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                # 检查子配置
                for subconfig_name in ['data', 'deeponet', 'training', 'physics', 'visualization', 'device']:
                    subconfig = getattr(self, subconfig_name)
                    if hasattr(subconfig, key):
                        setattr(subconfig, key, value)
                        break

    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            'data': self.data.__dict__,
            'deeponet': self.deeponet.__dict__,
            'training': self.training.__dict__,
            'physics': self.physics.__dict__,
            'visualization': self.visualization.__dict__,
            'device': self.device.__dict__
        }

    def save(self, filepath: str):
        """保存配置"""
        import json
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    def sync_probe_count(self):
        """同步探针数量配置：data.num_probes -> deeponet.probe_count -> deeponet.input_dim"""
        self.deeponet.probe_count = self.data.num_probes
        # 单Branch架构：探针数量 * 4 (x,y,real,imag) + 1 (频率)
        self.deeponet.input_dim = self.deeponet.probe_count * 4 + 1

    def print_config(self):
        """打印配置"""
        print("=== DeepONet电磁场重构配置 ===")
        for subconfig_name in ['data', 'deeponet', 'training', 'physics', 'visualization', 'device']:
            subconfig = getattr(self, subconfig_name)
            print(f"\n{subconfig_name.title()}:")
            for key, value in subconfig.__dict__.items():
                print(f"  {key}: {value}")
        print("=" * 50)


if __name__ == "__main__":
    # 测试配置
    cfg = Config()
    cfg.print_config()
    print(f"\n使用设备: {cfg.get_device()}")