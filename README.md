# 🔬 电磁场逆问题求解器 - 基于DeepONet的随机探针重建

> **Random Probe Electromagnetic Field Reconstruction using Single-Branch DeepONet**

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-11.8+-green.svg)](https://developer.nvidia.com/cuda-toolkit)

---

## 📋 目录

- [项目简介](#-项目简介)
- [核心特性](#-核心特性)
- [系统要求](#-系统要求)
- [快速开始](#-快速开始)
- [一键推理](#-一键推理客户盲测专用)
- [训练模型](#-训练模型可选)
- [项目结构](#-项目结构)
- [性能指标](#-性能指标)
- [常见问题](#-常见问题)

---

## 🎯 项目简介

本项目使用 **Single-Branch DeepONet** 架构，实现从**随机探针位置**的电磁场测量值重建整个空间的电磁场分布。

### 核心创新点

1. **🎲 随机探针位置**：每个样本使用独立的25个随机探针位置（非固定）
2. **🧠 Single-Branch架构**：101维输入（25×4+1），无需固定探针假设
3. **📊 2D k-space可视化**：插值到规则网格后进行2D FFT频谱分析
4. **⚡ 快速推理**：单样本推理 < 5秒，完整盲测集 < 30分钟

### 应用场景

- 电磁场反演重建
- 稀疏测量场重建
- 电磁兼容性(EMC)测试
- 天线近场诊断

---

## ✨ 核心特性

| 特性 | 说明 |
|------|------|
| **随机探针** | 25个探针位置每个样本独立随机，无需固定位置 |
| **高泛化能力** | 泛化比率 1.05× (Test/Train Loss) |
| **2D频谱分析** | 支持实空间 + k空间 FFT对比可视化 |
| **多频率支持** | 处理 2.3 - 13.0 GHz 频段 |
| **GPU加速** | 支持 CUDA 加速训练和推理 |
| **一键推理** | 自动化推理脚本，30分钟内完成盲测 |

---

## 💻 系统要求

### 硬件要求

- **GPU**: NVIDIA GPU with CUDA support (推荐 RTX 3090 / A100)
- **VRAM**: 至少 8GB
- **RAM**: 至少 16GB
- **存储**: 至少 5GB 可用空间

### 软件要求

- **操作系统**: Linux / macOS / Windows (推荐 Linux)
- **Python**: 3.11 或更高
- **CUDA**: 11.8 或更高 (如果使用GPU)

---

## 🚀 快速开始

### 步骤 1: 克隆项目

```bash
git clone https://github.com/Dopamine-mania/em_inverse.git
cd em_inverse
```

### 步骤 2: 创建虚拟环境

```bash
# 使用 conda (推荐)
conda create -n em_field python=3.11
conda activate em_field

# 或使用 venv
python -m venv venv
source venv/bin/activate  # Linux/macOS
# 或
.\venv\Scripts\activate  # Windows
```

### 步骤 3: 安装依赖

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install numpy pandas matplotlib scipy pyyaml tqdm
```

### 步骤 4: 验证安装

```bash
python -c "import torch; print(f'PyTorch版本: {torch.__version__}'); print(f'CUDA可用: {torch.cuda.is_available()}')"
```

预期输出：
```
PyTorch版本: 2.x.x
CUDA可用: True
```

---

## 🔮 一键推理（客户盲测专用）

### 使用说明

将客户提供的新CSV文件放入任意目录，然后运行：

```bash
python predict_new_data.py --input_dir /path/to/new/csv/files
```

### CSV文件格式要求

```csv
# Configuration 1 - Electromagnetic Field Data
X,Y,freq_1,Ez_real_1,Ez_imag_1,freq_2,Ez_real_2,Ez_imag_2
5.456951,-0.601177,5.051200,0.000000e+00,0.000000e+00,6.865134,0.000000e+00,0.000000e+00
...
```

- **第1行**: 注释行（自动跳过）
- **第2行**: 列名
- **后续行**: 数据（每行一个空间点）

### 输出结果

推理完成后，在 `outputs/blind_test/` 目录下生成：

```
outputs/blind_test/
├── real_space_*.png        # 实空间对比图（GT vs Pred vs Error）
├── kspace_*.png            # k空间频谱对比图（2D FFT）
└── inference_report.txt    # 详细推理报告
```

### 示例：完整流程

```bash
# 1. 将客户的CSV文件放入目录
mkdir -p customer_data
cp /path/to/customer/*.csv customer_data/

# 2. 运行一键推理
python predict_new_data.py --input_dir customer_data

# 3. 查看结果报告
cat outputs/blind_test/inference_report.txt

# 4. 查看可视化结果
ls outputs/blind_test/*.png
```

### 高级选项

```bash
# 自定义输出目录
python predict_new_data.py \
    --input_dir customer_data \
    --output_dir outputs/customer_results

# 指定不同的模型权重
python predict_new_data.py \
    --input_dir customer_data \
    --checkpoint_dir checkpoints/custom_model \
    --config config/custom_config.yaml
```

---

## 🎓 训练模型（可选）

如果需要重新训练模型或使用新数据集训练：

### 步骤 1: 准备数据

将CSV文件放入数据目录：

```bash
mkdir -p data/em_field_samples
cp /path/to/csv/files/*.csv data/em_field_samples/
```

### 步骤 2: 配置训练参数

编辑 `config/day2_fast_training.yaml`：

```yaml
data:
  data_path: "data/em_field_samples"
  num_probes: 25                      # 探针数量
  fixed_probe_positions: false        # 使用随机探针
  frequency_scale_factor: 1000.0      # 频率缩放因子

model:
  preset: "lightweight"               # 模型预设
  branch_hidden_dims: [256, 256]      # 隐藏层维度
  output_dim: 256                     # 输出维度
  activation: "gelu"                  # 激活函数

training:
  num_epochs: 1000                    # 训练轮数
  batch_size: 32                      # 批大小
  learning_rate: 0.0001               # 学习率
```

### 步骤 3: 开始训练

```bash
# 基础训练
python train.py --config config/day2_fast_training.yaml

# 从检查点恢复训练
python train.py \
    --config config/day2_fast_training.yaml \
    --resume checkpoints/day2_fast_training/epoch_0500.pth
```

### 步骤 4: 监控训练

```bash
# 在另一个终端运行监控脚本
./monitor_training.sh
```

### 步骤 5: 生成可视化

训练完成后，生成验证可视化：

```bash
python auto_visualize.py
```

---

## 📂 项目结构

```
em_inverse/
├── README.md                      # 本文档
├── .gitignore                     # Git忽略文件
├── predict_new_data.py            # 🔥 一键推理脚本（核心）
├── train.py                       # 训练脚本
├── auto_visualize.py              # 可视化生成脚本
├── auto_extend_training.sh        # 自动延长训练脚本
├── monitor_training.sh            # 训练监控脚本
│
├── config/                        # 配置文件
│   ├── day2_fast_training.yaml   # 生产配置
│   └── config.py                 # 配置加载器
│
├── model/                         # 模型定义
│   ├── enhanced_deeponet.py      # Single-Branch DeepONet
│   ├── enhanced_layers.py        # 增强层实现
│   └── ...
│
├── data/                          # 数据处理
│   └── dataset.py                # 数据集加载器
│
├── loss/                          # 损失函数
│   ├── loss.py                   # 组合损失
│   └── spectral_loss_gpu.py      # GPU加速频谱损失
│
├── checkpoints/                   # 模型权重
│   └── day2_fast_training/
│       └── best_epoch_0873_loss_4.217754.pth  # 最佳模型
│
└── outputs/                       # 输出结果
    ├── blind_test/               # 推理结果
    └── final_visualizations/      # 训练可视化
```

---

## 📊 性能指标

### 训练结果

| 指标 | 数值 |
|------|------|
| **最佳Epoch** | 873 |
| **Train Loss** | 4.01 |
| **Test Loss** | 4.18 |
| **泛化比率** | 1.05× (优秀) |
| **训练时间** | ~8小时 (1000 epochs, RTX 3090) |

### 推理性能

| 指标 | 数值 |
|------|------|
| **单样本推理** | < 5秒 |
| **批量推理 (100样本)** | < 3分钟 |
| **平均MSE** | 0.005 - 0.100 (取决于频率) |
| **GPU利用率** | > 90% |

### 测试集性能分布

- **最佳样本**: MSE = 0.0054
- **中等样本**: MSE = 0.0289
- **困难样本**: MSE = 0.1004
- **平均误差**: MSE = 0.0403

---

## ❓ 常见问题

### Q1: 如何判断推理结果好坏？

**A**: 查看以下指标：

- **MSE (总体)** < 0.1: 优秀
- **MSE (总体)** 0.1 - 0.3: 良好
- **MSE (总体)** > 0.3: 需要检查
- **Max Error** < 2.0: 优秀

### Q2: 可视化图中没有看到红色探针标记？

**A**: 确保使用最新版本的 `predict_new_data.py`。红色 × 标记应该清晰可见在实空间图上。

### Q3: 推理速度慢怎么办？

**A**: 检查以下几点：

```bash
# 1. 确认使用GPU
python -c "import torch; print(torch.cuda.is_available())"

# 2. 检查GPU使用情况
nvidia-smi

# 3. 减少batch size（如果内存不足）
# 编辑 predict_new_data.py 中的处理逻辑
```

### Q4: CSV文件格式不对怎么办？

**A**: 确保CSV文件符合以下格式：

1. 第一行是注释（以 # 开头）
2. 第二行是列名：`X,Y,freq_1,Ez_real_1,Ez_imag_1,freq_2,Ez_real_2,Ez_imag_2`
3. 后续每行是一个空间点的数据

### Q5: 如何添加自定义模型？

**A**:
1. 在 `model/` 目录下创建新的模型文件
2. 继承 `SingleBranchDeepONet` 类
3. 在 `config/*.yaml` 中配置新模型参数
4. 使用 `--config` 参数指定配置文件

### Q6: 训练时GPU内存不足？

**A**: 调整配置文件中的参数：

```yaml
training:
  batch_size: 16  # 从32降低到16

model:
  branch_hidden_dims: [128, 128]  # 从[256, 256]降低
```

### Q7: 如何查看详细的推理日志？

**A**: 推理脚本会自动生成详细报告：

```bash
cat outputs/blind_test/inference_report.txt
```

---

## 📞 技术支持

如遇到问题，请提供以下信息：

1. **错误信息截图**
2. **CSV文件格式示例**（前5行）
3. **运行环境信息**：
   ```bash
   python -c "import torch; print(f'Python: {__import__(\"sys\").version}'); print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
   ```

---

## 🎓 引用

如果使用本项目，请引用：

```bibtex
@software{em_inverse_deeponet,
  title = {Electromagnetic Field Reconstruction using Random Probe DeepONet},
  author = {Your Team},
  year = {2026},
  url = {https://github.com/Dopamine-mania/em_inverse}
}
```

---

## 📄 许可证

本项目遵循 MIT 许可证。详见 [LICENSE](LICENSE) 文件。

---

## 🙏 致谢

- DeepONet架构: Lu et al., 2021
- PyTorch框架: https://pytorch.org
- 数据提供: [客户名称]

---

<div align="center">

**⚡ 准备就绪，随时应对盲测挑战！⚡**

Made with ❤️ by [Your Team]

</div>
