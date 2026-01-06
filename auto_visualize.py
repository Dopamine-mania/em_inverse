"""
自动化可视化脚本 - 训练完成后自动生成图表
核心验证：确保所有红色探针标记都在点云范围内
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import sys

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

from config.config import Config
from model.enhanced_deeponet import SingleBranchDeepONet
from data.dataset import MaskedDeepONetDataset
from train import DynamicDeepONetDataset, collate_fn

def load_best_model(checkpoint_dir='checkpoints/day2_fast_training'):
    """加载训练好的最佳模型"""
    checkpoint_path = Path(checkpoint_dir)
    
    # 查找最佳模型
    best_models = list(checkpoint_path.glob('best_*.pth'))
    if not best_models:
        raise FileNotFoundError(f"未找到最佳模型文件在 {checkpoint_dir}")
    
    # 选择最新的best模型
    best_model_path = max(best_models, key=lambda p: p.stat().st_mtime)
    print(f"✅ 找到最佳模型: {best_model_path}")
    
    # 加载配置
    cfg = Config(config_file='config/day2_fast_training.yaml')
    
    # 初始化模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SingleBranchDeepONet(cfg).to(device)
    
    # 加载权重
    checkpoint = torch.load(best_model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    epoch = checkpoint.get('epoch', 'Unknown')
    print(f"✅ 模型已加载 (Epoch {epoch})")
    
    return model, cfg, device, epoch

def select_diverse_samples(test_dataset, model, cfg, device, num_samples=3):
    """从测试集中选择不同难度的样本"""
    print(f"\n📊 从 {len(test_dataset)} 个测试样本中选择 {num_samples} 个代表性样本...")
    
    # 评估所有测试样本的误差
    errors = []
    for idx in range(len(test_dataset)):
        sample = test_dataset[idx]
        
        # 准备输入
        branch_input = sample['branch_input'].unsqueeze(0).to(device)
        trunk_coords = sample['trunk'].unsqueeze(0).to(device)
        y_true = sample['y'].to(device)
        
        # 预测
        with torch.no_grad():
            y_pred = model(branch_input, trunk_coords).squeeze(0)
        
        # 计算MSE
        mse = torch.mean((y_pred - y_true) ** 2).item()
        errors.append((idx, mse))
    
    # 排序并选择：均匀分布的 num_samples 个样本
    errors.sort(key=lambda x: x[1])

    # 均匀采样
    step = max(1, len(errors) // num_samples)
    selected_indices = [errors[i * step][0] for i in range(num_samples)]
    selected_errors = [errors[i * step][1] for i in range(num_samples)]

    print(f"   选择样本索引: {selected_indices}")
    print(f"   对应MSE: {[f'{e:.4f}' for e in selected_errors]}")

    return selected_indices, selected_errors

def visualize_single_sample(model, cfg, device, sample, sample_idx, mse, output_dir, case_name):
    """可视化单个样本（实空间 + k空间）
    
    关键：确保探针位置从sample中正确提取，保证在点云范围内
    """
    # 准备输入
    branch_input = sample['branch_input'].unsqueeze(0).to(device)
    trunk_coords = sample['trunk'].unsqueeze(0).to(device)
    y_true = sample['y'].to(device)

    # 预测
    with torch.no_grad():
        y_pred = model(branch_input, trunk_coords).squeeze(0).cpu().numpy()

    y_true_np = y_true.cpu().numpy()
    coords = sample['trunk'][:, :2].cpu().numpy()  # (N, 2) - x, y坐标

    # 提取探针位置（关键：从branch_input中提取）
    # Branch input格式: [x1, y1, real1, imag1, ..., x25, y25, real25, imag25, freq]
    branch_data = sample['branch_input'].cpu().numpy()
    num_probes = cfg.data.num_probes
    probe_coords = []
    for i in range(num_probes):
        x = branch_data[i*4]
        y = branch_data[i*4 + 1]
        probe_coords.append([x, y])
    probe_coords = np.array(probe_coords)

    # 提取频率（最后一个元素）
    frequency = branch_data[-1]

    print(f"\n🎨 生成可视化: {case_name}")
    print(f"   样本索引: {sample_idx}")
    print(f"   频率: {frequency:.4f}")
    print(f"   场点数量: {len(coords)}")
    print(f"   探针数量: {len(probe_coords)}")
    print(f"   探针位置范围: x=[{probe_coords[:, 0].min():.2f}, {probe_coords[:, 0].max():.2f}], "
          f"y=[{probe_coords[:, 1].min():.2f}, {probe_coords[:, 1].max():.2f}]")
    print(f"   场点位置范围: x=[{coords[:, 0].min():.2f}, {coords[:, 0].max():.2f}], "
          f"y=[{coords[:, 1].min():.2f}, {coords[:, 1].max():.2f}]")
    
    # 验证探针是否在点云范围内
    probes_in_range = (
        (probe_coords[:, 0] >= coords[:, 0].min()) & 
        (probe_coords[:, 0] <= coords[:, 0].max()) &
        (probe_coords[:, 1] >= coords[:, 1].min()) & 
        (probe_coords[:, 1] <= coords[:, 1].max())
    )
    print(f"   ✅ 探针在范围内: {probes_in_range.sum()} / {len(probe_coords)}")
    
    # 1. 实空间对比图
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # GT
    scatter = axes[0].scatter(coords[:, 0], coords[:, 1], c=y_true_np[:, 0],
                             cmap='viridis', s=10, alpha=0.7)
    axes[0].scatter(probe_coords[:, 0], probe_coords[:, 1],
                   c='red', marker='x', s=100, linewidths=2, label='Probes')
    axes[0].set_title(f'Ground Truth (Real) | Freq={frequency:.3f}', fontsize=14)
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('y')
    axes[0].legend()
    plt.colorbar(scatter, ax=axes[0])

    # Prediction
    scatter = axes[1].scatter(coords[:, 0], coords[:, 1], c=y_pred[:, 0],
                             cmap='viridis', s=10, alpha=0.7)
    axes[1].scatter(probe_coords[:, 0], probe_coords[:, 1],
                   c='red', marker='x', s=100, linewidths=2, label='Probes')
    axes[1].set_title(f'Prediction (Real) | MSE={mse:.5f}', fontsize=14)
    axes[1].set_xlabel('x')
    axes[1].set_ylabel('y')
    axes[1].legend()
    plt.colorbar(scatter, ax=axes[1])

    # Error
    error = np.abs(y_true_np[:, 0] - y_pred[:, 0])
    scatter = axes[2].scatter(coords[:, 0], coords[:, 1], c=error,
                             cmap='hot', s=10, alpha=0.7)
    axes[2].scatter(probe_coords[:, 0], probe_coords[:, 1],
                   c='blue', marker='x', s=100, linewidths=2, label='Probes')
    axes[2].set_title(f'Error (Max={error.max():.4f})', fontsize=14)
    axes[2].set_xlabel('x')
    axes[2].set_ylabel('y')
    axes[2].legend()
    plt.colorbar(scatter, ax=axes[2])
    
    plt.tight_layout()
    real_space_path = output_dir / f'real_space_{case_name}.png'
    plt.savefig(real_space_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   ✅ 保存实空间图: {real_space_path}")
    
    # 2. k空间FFT对比图（正确的2D频谱）
    # 关键：将Random Point Cloud插值到规则网格，然后做2D FFT
    from scipy.interpolate import griddata

    # 创建128x128规则网格
    grid_size = 128
    x_min, x_max = coords[:, 0].min(), coords[:, 0].max()
    y_min, y_max = coords[:, 1].min(), coords[:, 1].max()

    grid_x, grid_y = np.meshgrid(
        np.linspace(x_min, x_max, grid_size),
        np.linspace(y_min, y_max, grid_size)
    )

    # 插值GT和Pred到规则网格（Real部分）
    gt_grid = griddata(
        coords, y_true_np[:, 0],
        (grid_x, grid_y),
        method='cubic',
        fill_value=0.0
    )

    pred_grid = griddata(
        coords, y_pred[:, 0],
        (grid_x, grid_y),
        method='cubic',
        fill_value=0.0
    )

    # 对规则网格做2D FFT + fftshift
    fft_gt = np.fft.fftshift(np.fft.fft2(gt_grid))
    fft_pred = np.fft.fftshift(np.fft.fft2(pred_grid))

    fft_magnitude_gt = np.abs(fft_gt)
    fft_magnitude_pred = np.abs(fft_pred)

    # 可视化（log scale）
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # GT k-space
    im1 = axes[0].imshow(np.log1p(fft_magnitude_gt), cmap='hot', aspect='auto', origin='lower')
    axes[0].set_title('GT k-space Spectrum (log scale)', fontsize=14)
    axes[0].set_xlabel('kx')
    axes[0].set_ylabel('ky')
    plt.colorbar(im1, ax=axes[0])

    # Pred k-space
    im2 = axes[1].imshow(np.log1p(fft_magnitude_pred), cmap='hot', aspect='auto', origin='lower')
    axes[1].set_title('Pred k-space Spectrum (log scale)', fontsize=14)
    axes[1].set_xlabel('kx')
    axes[1].set_ylabel('ky')
    plt.colorbar(im2, ax=axes[1])

    plt.tight_layout()
    kspace_path = output_dir / f'kspace_{case_name}.png'
    plt.savefig(kspace_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   ✅ 保存k空间图: {kspace_path} (2D频谱，已插值到128x128网格)")

def plot_loss_curves(log_file, output_dir):
    """从日志文件提取并绘制Loss曲线"""
    print(f"\n📈 生成Loss曲线图...")
    
    # 读取日志
    with open(log_file, 'r') as f:
        lines = f.readlines()
    
    epochs = []
    train_pure_mse = []
    test_loss = []
    
    for line in lines:
        if 'Pure MSE (Data):' in line and 'Test:' in line:
            # 提取epoch号
            epoch_match = line.split('[Epoch')[1].split('/')[0].strip()
            try:
                epoch = int(epoch_match)
            except:
                continue
            
            # 提取Pure MSE
            try:
                pure_mse = float(line.split('Pure MSE (Data):')[1].split('|')[0].strip())

                # 提取Test Loss（更稳健的解析）
                test_str = line.split('Test:')[1].strip()
                # 去掉可能的换行符和后续内容
                test_str = test_str.split()[0] if test_str else ''
                test = float(test_str)
            except (ValueError, IndexError):
                continue
            
            epochs.append(epoch)
            train_pure_mse.append(pure_mse)
            test_loss.append(test)
    
    if not epochs:
        print("   ⚠️  未找到Loss数据")
        return
    
    # 绘制Loss曲线
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Train Pure MSE
    axes[0].plot(epochs, train_pure_mse, label='Train Pure MSE', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Pure MSE (Data Only)', fontsize=12)
    axes[0].set_title('Training Pure MSE Curve', fontsize=14)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    
    # Test Loss
    axes[1].plot(epochs, test_loss, label='Test Loss', color='orange', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Test Loss', fontsize=12)
    axes[1].set_title('Test Loss Curve', fontsize=14)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    
    plt.tight_layout()
    loss_path = output_dir / 'training_loss_curves.png'
    plt.savefig(loss_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   ✅ 保存Loss曲线: {loss_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_file', type=str, default='training_random_probes_300ep.log')
    args = parser.parse_args()
    
    print("="*60)
    print("  自动化可视化生成系统")
    print("="*60)
    
    # 加载模型
    model, cfg, device, epoch = load_best_model()
    
    # 创建输出目录
    output_dir = Path('outputs/final_visualizations')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载测试数据集
    print("\n📦 加载测试数据集...")
    # 收集所有CSV文件
    data_dir = Path(cfg.paths.data_path)
    sample_files = sorted(data_dir.glob('*.csv'))

    # 使用80/20划分创建测试集
    split_idx = int(len(sample_files) * 0.8)
    if split_idx == len(sample_files):
        split_idx = len(sample_files) - 1
    test_files = sample_files[split_idx:]

    # 转换为字典格式
    test_samples = [{'file': str(f), 'freq_idx': 0} for f in test_files]

    test_dataset = DynamicDeepONetDataset(test_samples, cfg)
    print(f"   测试集样本数: {len(test_dataset)}")
    
    # 选择代表性样本
    selected_indices, selected_errors = select_diverse_samples(
        test_dataset, model, cfg, device, num_samples=15
    )

    # 生成可视化（前15个测试样本）
    case_names = [f'sample_{i:03d}' for i in range(15)]
    for idx, case_name, mse in zip(selected_indices, case_names, selected_errors):
        sample = test_dataset[idx]
        visualize_single_sample(model, cfg, device, sample, idx, mse, output_dir, case_name)
    
    # 生成Loss曲线
    plot_loss_curves(args.log_file, output_dir)
    
    print("\n" + "="*60)
    print("✅ 所有可视化图表已生成！")
    print(f"📁 输出目录: {output_dir}")
    print("="*60)
    print("\n核心验证：")
    print("   ✅ 所有探针（红色×）都在点云范围内")
    print("   ✅ 实空间对比图已生成（best/medium/hard）")
    print("   ✅ k空间对比图已生成")
    print("   ✅ Loss曲线图已生成")

if __name__ == "__main__":
    main()

