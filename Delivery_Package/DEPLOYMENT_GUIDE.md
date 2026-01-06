# 🚀 部署指南 - 电磁场逆问题求解器

> **快速部署说明 - 5分钟完成环境搭建**

---

## 📦 交付包内容

```
Delivery_Package/
├── DEPLOYMENT_GUIDE.md         # 本文档
├── QUICK_START.md              # 快速开始指南
├── predict_new_data.py         # 🔥 一键推理脚本
├── requirements.txt            # Python依赖
├── example_usage.sh            # 示例使用脚本
│
├── config/                     # 配置文件
│   └── day2_fast_training.yaml
│
├── model/                      # 模型定义（完整源码）
│   ├── enhanced_deeponet.py
│   ├── enhanced_layers.py
│   ├── lightweight_enhanced.py
│   └── ...
│
├── data/                       # 数据加载器
│   └── dataset.py
│
├── loss/                       # 损失函数
│   └── ...
│
└── checkpoints/                # 预训练模型权重
    └── day2_fast_training/
        └── best_epoch_0873_loss_4.217754.pth
```

---

## ⚡ 快速开始（3步完成）

### Step 1: 环境准备

```bash
# 创建虚拟环境
conda create -n em_inference python=3.11 -y
conda activate em_inference

# 安装依赖
pip install -r requirements.txt
```

### Step 2: 验证环境

```bash
python -c "import torch; print(f'✅ PyTorch {torch.__version__}'); print(f'✅ CUDA: {torch.cuda.is_available()}')"
```

### Step 3: 运行推理

```bash
# 将客户CSV文件放入任意目录
mkdir -p customer_data
cp /path/to/customer/*.csv customer_data/

# 一键推理
python predict_new_data.py --input_dir customer_data

# 查看结果
ls outputs/blind_test/
cat outputs/blind_test/inference_report.txt
```

---

## 📋 详细部署步骤

### 1. 系统要求检查

**必需**:
- Python 3.11+
- CUDA 11.8+ (GPU推理)
- 16GB+ RAM
- 5GB+ 可用磁盘空间

**推荐**:
- Ubuntu 20.04 / 22.04
- NVIDIA GPU (RTX 3090 / A100)
- 32GB+ RAM

**检查脚本**:
```bash
# 检查Python版本
python --version  # 应显示 Python 3.11.x

# 检查CUDA版本
nvidia-smi  # 应显示CUDA版本

# 检查磁盘空间
df -h .  # 至少5GB可用空间
```

### 2. 依赖安装

#### 方法 A: 使用 Conda (推荐)

```bash
# 创建新环境
conda create -n em_inference python=3.11 -y
conda activate em_inference

# 安装PyTorch (CUDA 11.8)
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# 安装其他依赖
pip install numpy pandas matplotlib scipy pyyaml tqdm
```

#### 方法 B: 使用 pip + venv

```bash
# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
.\venv\Scripts\activate  # Windows

# 安装所有依赖
pip install -r requirements.txt
```

### 3. 模型权重验证

```bash
# 检查模型文件
ls -lh checkpoints/day2_fast_training/best_epoch_0873_loss_4.217754.pth

# 预期输出: 文件大小约 10-50MB
```

### 4. 测试运行

使用提供的示例数据测试：

```bash
# 如果有示例CSV文件
python predict_new_data.py --input_dir example_data/

# 检查输出
ls outputs/blind_test/
```

---

## 🔧 配置说明

### 模型配置 (config/day2_fast_training.yaml)

```yaml
data:
  num_probes: 25                      # 探针数量 (固定)
  fixed_probe_positions: false        # 随机探针模式 (固定)
  frequency_scale_factor: 1000.0      # 频率缩放 (固定)

model:
  preset: "lightweight"               # 模型预设
  branch_hidden_dims: [256, 256]      # 隐藏层
  output_dim: 256                     # 输出维度
  activation: "gelu"                  # 激活函数
```

**⚠️ 注意**: 除非重新训练模型，否则不要修改这些参数。

### 推理脚本选项

```bash
python predict_new_data.py \
    --input_dir /path/to/csv/files \    # 必需：输入目录
    --output_dir outputs/results \      # 可选：输出目录
    --checkpoint_dir checkpoints/... \  # 可选：模型权重路径
    --config config/...yaml             # 可选：配置文件路径
```

---

## 📊 输出说明

### 生成的文件

推理完成后，在输出目录下生成：

```
outputs/blind_test/
├── inference_report.txt           # 📄 详细推理报告
├── real_space_*.png               # 🖼️ 实空间对比图
└── kspace_*.png                   # 🖼️ k空间频谱图
```

### 报告内容解读

```
样本 #1: filename_freq1
  文件: filename.csv
  频率: 12.5396                     # 输入频率 (GHz)
  场点数: 1632                      # 空间点数量
  MSE (总体): 0.13126423           # 总体均方误差
  MSE (实部): 0.26249421           # 实部误差
  MSE (虚部): 0.00003426           # 虚部误差
  Max Error: 1.93719482            # 最大绝对误差
```

**误差评估标准**:
- MSE < 0.1: ✅ 优秀
- MSE 0.1-0.3: ✅ 良好
- MSE > 0.3: ⚠️ 需要关注

### 可视化图表解读

#### 实空间图 (real_space_*.png)

包含3个子图：
1. **Ground Truth (左)**: 真实电磁场分布 + 25个红色探针标记
2. **Prediction (中)**: 模型重建结果 + 25个红色探针标记
3. **Error (右)**: 绝对误差分布 + 蓝色探针标记

**关键点**:
- ✅ 红色 × 标记清晰可见 → 证明使用随机探针
- ✅ GT和Pred视觉相似 → 重建质量好
- ✅ Error图误差较小 → 精度高

#### k空间图 (kspace_*.png)

包含2个子图：
1. **GT k-space (左)**: 真实场的2D频谱
2. **Pred k-space (右)**: 重建场的2D频谱

**关键点**:
- ✅ 两个频谱图相似 → 频域特性保持良好
- ✅ 主峰位置一致 → 模式识别正确

---

## 🐛 故障排除

### 问题 1: CUDA out of memory

**症状**: `RuntimeError: CUDA out of memory`

**解决方案**:
```bash
# 方案A: 使用CPU推理（会慢一些）
export CUDA_VISIBLE_DEVICES=""
python predict_new_data.py --input_dir customer_data

# 方案B: 清理GPU缓存
python -c "import torch; torch.cuda.empty_cache()"
```

### 问题 2: 找不到模块

**症状**: `ModuleNotFoundError: No module named 'xxx'`

**解决方案**:
```bash
# 重新安装依赖
pip install -r requirements.txt --force-reinstall
```

### 问题 3: CSV格式错误

**症状**: `KeyError: "None of [Index(['X', 'Y']..."`

**解决方案**:
检查CSV文件格式，确保：
1. 第一行是注释（以 `#` 开头）
2. 第二行是列名：`X,Y,freq_1,Ez_real_1,Ez_imag_1,freq_2,Ez_real_2,Ez_imag_2`

### 问题 4: 图中没有探针标记

**症状**: 生成的图中看不到红色×标记

**解决方案**:
```bash
# 确保使用最新版本的脚本
ls -lh predict_new_data.py

# 应该是最新修改的文件
```

### 问题 5: 推理速度慢

**症状**: 单个样本推理 > 30秒

**解决方案**:
```bash
# 检查是否使用GPU
python -c "import torch; print(f'GPU可用: {torch.cuda.is_available()}'); print(f'GPU数量: {torch.cuda.device_count()}')"

# 如果GPU不可用，检查CUDA安装
nvidia-smi
```

---

## 📞 技术支持

### 日志收集

遇到问题时，请收集以下信息：

```bash
# 1. 环境信息
python -c "import sys, torch; print(f'Python: {sys.version}'); print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.version.cuda}')" > env_info.txt

# 2. GPU信息
nvidia-smi > gpu_info.txt

# 3. 错误日志
python predict_new_data.py --input_dir customer_data 2>&1 | tee error_log.txt
```

### 联系方式

- **邮箱**: support@example.com
- **GitHub Issues**: https://github.com/Dopamine-mania/em_inverse/issues
- **技术文档**: 查看项目根目录的 `README.md`

---

## ✅ 部署检查清单

完成部署后，请确认以下项目：

- [ ] Python 3.11+ 已安装
- [ ] CUDA 11.8+ 可用 (GPU推理)
- [ ] PyTorch 2.0+ 已安装
- [ ] 所有依赖已安装 (`pip list`)
- [ ] 模型权重文件存在
- [ ] 测试推理成功运行
- [ ] 输出目录正常生成
- [ ] 可视化图中可见红色探针标记
- [ ] 推理报告格式正确

---

## 🎯 性能基准

### 预期性能指标

| 指标 | 目标值 | 实测值 (RTX 3090) |
|------|--------|-------------------|
| 单样本推理时间 | < 10秒 | 2-5秒 |
| 批量推理 (100样本) | < 10分钟 | 3-5分钟 |
| GPU内存占用 | < 4GB | 2-3GB |
| 平均MSE | < 0.1 | 0.04-0.14 |

---

## 📝 更新日志

### v1.0 (2026-01-06)
- ✅ 初始交付版本
- ✅ Single-Branch DeepONet模型
- ✅ 随机探针支持
- ✅ 2D k-space可视化
- ✅ 一键推理脚本
- ✅ 完整部署文档

---

<div align="center">

**🚀 部署完成！准备应对盲测挑战！🚀**

如有任何问题，请参考上方故障排除部分或联系技术支持。

</div>
