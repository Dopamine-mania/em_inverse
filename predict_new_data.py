"""
🚀 一键推理脚本 - 客户盲测专用
====================================

用法：
    python predict_new_data.py --input_dir /path/to/new/csv/files

功能：
    1. 自动加载最佳训练模型
    2. 读取新的CSV文件（客户盲测数据）
    3. 对每个样本生成可视化（实空间 + k空间）
    4. 显示25个红色探针标记
    5. 计算并报告 MSE 和 Max Error
    6. 30分钟内完成所有推理和可视化

输出：
    - outputs/blind_test/real_space_*.png
    - outputs/blind_test/kspace_*.png
    - outputs/blind_test/inference_report.txt
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import sys
from datetime import datetime
from scipy.interpolate import griddata

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

from config.config import Config
from model.enhanced_deeponet import SingleBranchDeepONet

def load_best_model(checkpoint_dir='checkpoints/day2_fast_training', config_file='config/day2_fast_training.yaml'):
    """加载训练好的最佳模型"""
    print("=" * 80)
    print("🔧 正在加载模型...")
    print("=" * 80)

    checkpoint_path = Path(checkpoint_dir)

    # 查找最佳模型
    best_models = list(checkpoint_path.glob('best_*.pth'))
    if not best_models:
        raise FileNotFoundError(f"❌ 未找到最佳模型文件在 {checkpoint_dir}")

    # 选择最新的best模型
    best_model_path = max(best_models, key=lambda p: p.stat().st_mtime)
    print(f"✅ 找到最佳模型: {best_model_path.name}")

    # 加载配置
    cfg = Config(config_file=config_file)
    print(f"✅ 加载配置: {config_file}")

    # 初始化模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"✅ 使用设备: {device}")

    model = SingleBranchDeepONet(cfg).to(device)

    # 加载权重
    checkpoint = torch.load(best_model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    epoch = checkpoint.get('epoch', 'Unknown')
    train_loss = checkpoint.get('train_loss', 'Unknown')
    test_loss = checkpoint.get('test_loss', 'Unknown')

    print(f"✅ 模型已加载:")
    print(f"   - Epoch: {epoch}")
    print(f"   - Train Loss: {train_loss}")
    print(f"   - Test Loss: {test_loss}")
    print()

    return model, cfg, device

def load_new_csv(csv_path, cfg):
    """加载新的CSV文件并准备数据

    CSV格式：
        X, Y, freq_1, Ez_real_1, Ez_imag_1, freq_2, Ez_real_2, Ez_imag_2

    返回：
        两个样本的列表（每个频率一个）
    """
    # 跳过第一行注释
    df = pd.read_csv(csv_path, skiprows=1)

    # 提取坐标
    coords = df[['X', 'Y']].values  # (N, 2)

    # 提取两个频率的数据
    samples = []

    for freq_idx in [1, 2]:
        freq_col = f'freq_{freq_idx}'
        real_col = f'Ez_real_{freq_idx}'
        imag_col = f'Ez_imag_{freq_idx}'

        if freq_col not in df.columns:
            print(f"⚠️  警告: CSV中没有找到 {freq_col} 列，跳过")
            continue

        frequency = df[freq_col].iloc[0]  # 假设整列相同
        Ez_real = df[real_col].values
        Ez_imag = df[imag_col].values

        samples.append({
            'coords': coords,
            'frequency': frequency,
            'Ez_real': Ez_real,
            'Ez_imag': Ez_imag,
            'csv_name': Path(csv_path).stem,
            'freq_idx': freq_idx
        })

    return samples

def prepare_model_input(sample, cfg, device):
    """准备模型输入

    流程：
        1. 随机选择25个探针位置
        2. 构建101维 branch input
        3. 构建 trunk input (所有坐标)
        4. 构建 ground truth
    """
    coords = sample['coords']  # (N, 2)
    Ez_real = sample['Ez_real']  # (N,)
    Ez_imag = sample['Ez_imag']  # (N,)
    frequency = sample['frequency']

    num_points = len(coords)
    num_probes = cfg.data.num_probes

    # 随机选择25个探针
    np.random.seed(42)  # 固定随机种子以便复现
    probe_indices = np.random.choice(num_points, num_probes, replace=False)

    # 构建 branch input (101维)
    branch_data = []
    for idx in probe_indices:
        x, y = coords[idx]
        real = Ez_real[idx]
        imag = Ez_imag[idx]
        branch_data.extend([x, y, real, imag])

    # 添加频率（归一化）
    normalized_freq = frequency / cfg.data.frequency_scale_factor
    branch_data.append(normalized_freq)

    branch_input = torch.tensor(branch_data, dtype=torch.float32).unsqueeze(0).to(device)  # (1, 101)

    # 构建 trunk input (所有坐标 + 频率)
    # 模型期望 (batch_size, num_points, 3) 其中 3 = [x, y, freq]
    trunk_coords_with_freq = np.concatenate([
        coords,
        np.full((num_points, 1), normalized_freq)
    ], axis=1)  # (N, 3)
    trunk_input = torch.tensor(trunk_coords_with_freq, dtype=torch.float32).unsqueeze(0).to(device)  # (1, N, 3)

    # Ground truth (实部和虚部)
    y_true = np.stack([Ez_real, Ez_imag], axis=1)  # (N, 2)

    # 提取探针坐标
    probe_coords = coords[probe_indices]

    return branch_input, trunk_input, y_true, probe_coords, probe_indices

def predict_sample(model, sample, cfg, device):
    """对单个样本进行推理"""
    branch_input, trunk_input, y_true, probe_coords, probe_indices = prepare_model_input(sample, cfg, device)

    # 推理
    with torch.no_grad():
        y_pred = model(branch_input, trunk_input).squeeze(0).cpu().numpy()  # (N, 2)

    # 计算误差
    mse = np.mean((y_pred - y_true) ** 2)
    max_error = np.max(np.abs(y_pred - y_true))

    # 分别计算实部和虚部的误差
    mse_real = np.mean((y_pred[:, 0] - y_true[:, 0]) ** 2)
    mse_imag = np.mean((y_pred[:, 1] - y_true[:, 1]) ** 2)

    return {
        'y_pred': y_pred,
        'y_true': y_true,
        'coords': sample['coords'],
        'probe_coords': probe_coords,
        'frequency': sample['frequency'],
        'mse': mse,
        'mse_real': mse_real,
        'mse_imag': mse_imag,
        'max_error': max_error
    }

def visualize_results(result, output_dir, case_name):
    """生成可视化图表（实空间 + k空间）"""
    coords = result['coords']
    y_true = result['y_true']
    y_pred = result['y_pred']
    probe_coords = result['probe_coords']
    frequency = result['frequency']
    mse = result['mse']
    max_error = result['max_error']

    # ============ 1. 实空间对比图 ============
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # GT (Real part)
    scatter = axes[0].scatter(coords[:, 0], coords[:, 1], c=y_true[:, 0],
                             cmap='viridis', s=10, alpha=0.7)
    axes[0].scatter(probe_coords[:, 0], probe_coords[:, 1],
                   c='red', marker='x', s=150, linewidths=3,
                   label='25 Random Probes', zorder=10)
    axes[0].set_title(f'Ground Truth (Real) | Freq={frequency:.3f}', fontsize=16)
    axes[0].set_xlabel('x', fontsize=12)
    axes[0].set_ylabel('y', fontsize=12)
    axes[0].legend(fontsize=10)
    plt.colorbar(scatter, ax=axes[0])

    # Prediction (Real part)
    scatter = axes[1].scatter(coords[:, 0], coords[:, 1], c=y_pred[:, 0],
                             cmap='viridis', s=10, alpha=0.7)
    axes[1].scatter(probe_coords[:, 0], probe_coords[:, 1],
                   c='red', marker='x', s=150, linewidths=3,
                   label='25 Random Probes', zorder=10)
    axes[1].set_title(f'Prediction (Real) | MSE={mse:.6f}', fontsize=16)
    axes[1].set_xlabel('x', fontsize=12)
    axes[1].set_ylabel('y', fontsize=12)
    axes[1].legend(fontsize=10)
    plt.colorbar(scatter, ax=axes[1])

    # Error
    error = np.abs(y_true[:, 0] - y_pred[:, 0])
    scatter = axes[2].scatter(coords[:, 0], coords[:, 1], c=error,
                             cmap='hot', s=10, alpha=0.7)
    axes[2].scatter(probe_coords[:, 0], probe_coords[:, 1],
                   c='blue', marker='x', s=150, linewidths=3,
                   label='Probes', zorder=10)
    axes[2].set_title(f'Error (Max={max_error:.6f})', fontsize=16)
    axes[2].set_xlabel('x', fontsize=12)
    axes[2].set_ylabel('y', fontsize=12)
    axes[2].legend(fontsize=10)
    plt.colorbar(scatter, ax=axes[2])

    plt.tight_layout()
    real_space_path = output_dir / f'real_space_{case_name}.png'
    plt.savefig(real_space_path, dpi=150, bbox_inches='tight')
    plt.close()

    # ============ 2. k空间FFT对比图 ============
    # 插值到128x128规则网格
    grid_size = 128
    x_min, x_max = coords[:, 0].min(), coords[:, 0].max()
    y_min, y_max = coords[:, 1].min(), coords[:, 1].max()

    grid_x, grid_y = np.meshgrid(
        np.linspace(x_min, x_max, grid_size),
        np.linspace(y_min, y_max, grid_size)
    )

    # 插值GT和Pred到规则网格（Real部分）
    gt_grid = griddata(
        coords, y_true[:, 0],
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

    # 2D FFT + fftshift
    fft_gt = np.fft.fftshift(np.fft.fft2(gt_grid))
    fft_pred = np.fft.fftshift(np.fft.fft2(pred_grid))

    fft_magnitude_gt = np.abs(fft_gt)
    fft_magnitude_pred = np.abs(fft_pred)

    # 可视化（log scale）
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # GT k-space
    im1 = axes[0].imshow(np.log1p(fft_magnitude_gt), cmap='hot', aspect='auto', origin='lower')
    axes[0].set_title(f'GT k-space Spectrum | Freq={frequency:.3f}', fontsize=14)
    axes[0].set_xlabel('kx', fontsize=12)
    axes[0].set_ylabel('ky', fontsize=12)
    plt.colorbar(im1, ax=axes[0])

    # Pred k-space
    im2 = axes[1].imshow(np.log1p(fft_magnitude_pred), cmap='hot', aspect='auto', origin='lower')
    axes[1].set_title(f'Pred k-space Spectrum | MSE={mse:.6f}', fontsize=14)
    axes[1].set_xlabel('kx', fontsize=12)
    axes[1].set_ylabel('ky', fontsize=12)
    plt.colorbar(im2, ax=axes[1])

    plt.tight_layout()
    kspace_path = output_dir / f'kspace_{case_name}.png'
    plt.savefig(kspace_path, dpi=150, bbox_inches='tight')
    plt.close()

    return real_space_path, kspace_path

def main():
    parser = argparse.ArgumentParser(description='一键推理脚本 - 客户盲测专用')
    parser.add_argument('--input_dir', type=str, required=True,
                        help='新CSV文件所在目录')
    parser.add_argument('--output_dir', type=str, default='outputs/blind_test',
                        help='输出目录（默认: outputs/blind_test）')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints/day2_fast_training',
                        help='模型权重目录（默认: checkpoints/day2_fast_training）')
    parser.add_argument('--config', type=str, default='config/day2_fast_training.yaml',
                        help='配置文件（默认: config/day2_fast_training.yaml）')

    args = parser.parse_args()

    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 开始计时
    start_time = datetime.now()

    print("\n" + "=" * 80)
    print("🚀 一键推理系统 - 客户盲测专用")
    print("=" * 80)
    print(f"⏰ 开始时间: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📂 输入目录: {args.input_dir}")
    print(f"📁 输出目录: {output_dir}")
    print()

    # 加载模型
    model, cfg, device = load_best_model(args.checkpoint_dir, args.config)

    # 查找所有CSV文件
    input_path = Path(args.input_dir)
    csv_files = sorted(input_path.glob('*.csv'))

    if not csv_files:
        print(f"❌ 错误: 在 {input_path} 中未找到CSV文件")
        return

    print(f"📊 找到 {len(csv_files)} 个CSV文件")
    print()

    # 准备报告
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("📈 推理结果报告")
    report_lines.append("=" * 80)
    report_lines.append(f"时间: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append(f"模型: {args.checkpoint_dir}")
    report_lines.append(f"输入目录: {args.input_dir}")
    report_lines.append(f"样本数量: {len(csv_files) * 2} (每个CSV包含2个频率)")
    report_lines.append("")

    # 逐个处理CSV文件
    total_samples = 0
    all_results = []

    for csv_idx, csv_file in enumerate(csv_files, 1):
        print(f"{'=' * 80}")
        print(f"📄 处理文件 [{csv_idx}/{len(csv_files)}]: {csv_file.name}")
        print(f"{'=' * 80}")

        # 加载CSV
        samples = load_new_csv(csv_file, cfg)
        print(f"   ✅ 加载完成，包含 {len(samples)} 个样本（2个频率）")

        # 对每个样本进行推理
        for sample_idx, sample in enumerate(samples, 1):
            total_samples += 1
            case_name = f"{csv_file.stem}_freq{sample['freq_idx']}"

            print(f"\n   🔮 样本 {sample_idx}/{len(samples)}: freq={sample['frequency']:.4f}")
            print(f"      - 场点数量: {len(sample['coords'])}")
            print(f"      - 探针数量: {cfg.data.num_probes}")

            # 推理
            result = predict_sample(model, sample, cfg, device)

            print(f"      ✅ 推理完成:")
            print(f"         MSE (总体): {result['mse']:.8f}")
            print(f"         MSE (实部): {result['mse_real']:.8f}")
            print(f"         MSE (虚部): {result['mse_imag']:.8f}")
            print(f"         Max Error: {result['max_error']:.8f}")

            # 生成可视化
            real_path, kspace_path = visualize_results(result, output_dir, case_name)
            print(f"      ✅ 可视化已保存:")
            print(f"         - {real_path.name}")
            print(f"         - {kspace_path.name}")

            # 记录到报告
            report_lines.append(f"样本 #{total_samples}: {case_name}")
            report_lines.append(f"  文件: {csv_file.name}")
            report_lines.append(f"  频率: {sample['frequency']:.4f}")
            report_lines.append(f"  场点数: {len(sample['coords'])}")
            report_lines.append(f"  MSE (总体): {result['mse']:.8f}")
            report_lines.append(f"  MSE (实部): {result['mse_real']:.8f}")
            report_lines.append(f"  MSE (虚部): {result['mse_imag']:.8f}")
            report_lines.append(f"  Max Error: {result['max_error']:.8f}")
            report_lines.append("")

            all_results.append({
                'case_name': case_name,
                'csv_file': csv_file.name,
                'frequency': sample['frequency'],
                'mse': result['mse'],
                'mse_real': result['mse_real'],
                'mse_imag': result['mse_imag'],
                'max_error': result['max_error']
            })

    # 计算统计信息
    avg_mse = np.mean([r['mse'] for r in all_results])
    avg_mse_real = np.mean([r['mse_real'] for r in all_results])
    avg_mse_imag = np.mean([r['mse_imag'] for r in all_results])
    avg_max_error = np.mean([r['max_error'] for r in all_results])

    report_lines.append("=" * 80)
    report_lines.append("📊 统计汇总")
    report_lines.append("=" * 80)
    report_lines.append(f"总样本数: {total_samples}")
    report_lines.append(f"平均 MSE (总体): {avg_mse:.8f}")
    report_lines.append(f"平均 MSE (实部): {avg_mse_real:.8f}")
    report_lines.append(f"平均 MSE (虚部): {avg_mse_imag:.8f}")
    report_lines.append(f"平均 Max Error: {avg_max_error:.8f}")

    # 结束计时
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()

    report_lines.append("")
    report_lines.append(f"⏱️  总耗时: {duration:.2f} 秒")
    report_lines.append(f"✅ 所有推理完成！")
    report_lines.append("=" * 80)

    # 保存报告
    report_path = output_dir / 'inference_report.txt'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))

    # 打印最终报告
    print("\n" + "=" * 80)
    print("🎉 推理完成！")
    print("=" * 80)
    print(f"✅ 处理文件数: {len(csv_files)}")
    print(f"✅ 总样本数: {total_samples}")
    print(f"✅ 平均 MSE: {avg_mse:.8f}")
    print(f"✅ 平均 Max Error: {avg_max_error:.8f}")
    print(f"⏱️  总耗时: {duration:.2f} 秒")
    print(f"📁 输出目录: {output_dir}")
    print(f"📄 详细报告: {report_path}")
    print("=" * 80)

    # 如果超过30分钟，发出警告
    if duration > 1800:
        print("⚠️  警告: 推理耗时超过30分钟！")
    else:
        print(f"✅ 推理在 {duration/60:.1f} 分钟内完成，符合30分钟要求")

    print("\n🚀 准备就绪！可以立即发送给客户！")

if __name__ == "__main__":
    main()
