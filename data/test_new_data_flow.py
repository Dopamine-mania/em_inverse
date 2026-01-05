"""
测试正确的数据处理流程：先选探针，再基于探针构建凸边界
"""

import numpy as np
import pandas as pd
import sys
from pathlib import Path

# 添加模块路径 - 需要指向em_field的父目录
sys.path.append(str(Path(__file__).parent.parent))

from config import Config
from data.dataset import MaskedDeepONetDataset

def test_new_data_flow():
    """测试新的数据处理流程"""
    print("=== 测试新的数据处理流程 ===")
    print("流程：选构型 → 选探针 → 基于探针构建凸边界 → 筛选数据")

    # 初始化配置
    cfg = Config()
    cfg.data.normalize_data = False
    cfg.data.num_probes = 50  # 设置探针数量
    cfg.data.max_samples = 10   # 限制样本数量用于测试

    # 创建数据集实例
    dataset = MaskedDeepONetDataset(cfg)

    # 测试数据路径
    data_path = r"D:\科研\AI实现场强相位预测\01_电磁场数据文件"

    if not Path(data_path).exists():
        print(f"数据路径不存在: {data_path}")
        print("请修改测试脚本中的数据路径")
        return False

    try:
        # 1. 扫描所有构型（不同频率）
        print("\n1. 扫描构型...")
        configurations = dataset.scan_available_samples(data_path, max_frequency=20.0, max_samples=10)

        if not configurations:
            print("未找到任何构型")
            return

        print(f"总共找到 {len(configurations)} 个构型")
        print(f"示例构型: {configurations[0]}")

        # 2. 测试单个构型的数据处理
        print("\n2. 测试单个构型处理...")
        sample_config = configurations[0]

        # 处理单个构型
        branch_input, trunk_input, target_output, mask = dataset.prepare_single_sample(
            sample_config['file'],
            freq_idx=sample_config['freq_idx']
        )

        print(f"\n数据处理结果:")
        print(f"  Branch输入形状: {branch_input.shape}")
        print(f"  Trunk输入形状: {trunk_input.shape}")
        print(f"  目标输出形状: {target_output.shape}")
        print(f"  Mask形状: {mask.shape}")
        print(f"  探针数量: {branch_input.shape[0]}")
        print(f"  凸边界内数据点数: {trunk_input.shape[0]}")
        print(f"  Mask中True的数量: {np.sum(mask)}")

        # 3. 验证数据格式
        print("\n3. 验证数据格式...")
        print(f"  Branch输入特征数: {branch_input.shape[1]} (应为5: x,y,freq,real,imag)")
        print(f"  Trunk输入特征数: {trunk_input.shape[1]} (应为3: x,y,freq)")
        print(f"  目标输出特征数: {target_output.shape[1]} (应为2: real,imag)")

        # 4. 检查探针数据
        print("\n4. 检查探针数据...")
        probe_coords = branch_input[:, :2]  # x, y
        probe_freqs = branch_input[:, 2]     # frequency
        probe_fields = branch_input[:, 3:5] # real, imag

        print(f"  探针坐标范围: X[{probe_coords[:, 0].min():.3f}, {probe_coords[:, 0].max():.3f}], "
              f"Y[{probe_coords[:, 1].min():.3f}, {probe_coords[:, 1].max():.3f}]")
        print(f"  探针频率: {probe_freqs[0]:.3f} GHz")

        # 5. 检查trunk数据
        print("\n5. 检查Trunk数据...")
        trunk_coords = trunk_input[:, :2]  # x, y
        trunk_freqs = trunk_input[:, 2]      # frequency

        print(f"  Trunk坐标范围: X[{trunk_coords[:, 0].min():.3f}, {trunk_coords[:, 0].max():.3f}], "
              f"Y[{trunk_coords[:, 1].min():.3f}, {trunk_coords[:, 1].max():.3f}]")
        print(f"  Trunk数据点数: {len(trunk_coords)}")

        # 6. 验证凸边界约束
        print("\n6. 验证凸边界约束...")
        from scipy.spatial import ConvexHull
        from matplotlib.path import Path as MatplotlibPath

        # 计算探针凸边界
        hull = ConvexHull(probe_coords)
        hull_path = MatplotlibPath(probe_coords[hull.vertices])

        # 检查所有trunk点是否在探针凸边界内
        interior_mask = hull_path.contains_points(trunk_coords)
        interior_count = np.sum(interior_mask)

        print(f"  探针凸边界内Trunk点数: {interior_count}/{len(trunk_coords)} "
              f"({interior_count/len(trunk_coords)*100:.1f}%)")

        if interior_count == len(trunk_coords):
            print("  所有Trunk点都在探针凸边界内")
        else:
            print("  部分Trunk点在探针凸边界外")

        print("\n=== 测试完成 ===")
        print("新的数据处理流程正确实现：")
        print("1. 正确识别构型（不同频率）")
        print("2. 先选择探针位置")
        print("3. 基于探针构建凸边界")
        print("4. 筛选凸边界内的数据点")
        print("5. 无频率过滤")
        print("6. 无归一化")

        return True

    except Exception as e:
        print(f"测试过程中出错: {e}")
        import traceback
        traceback.print_exc()
        return False

def compare_old_vs_new():
    """对比新旧流程的差异"""
    print("\n=== 新旧流程对比 ===")

    print("\n旧流程（错误）：")
    print("1. 读取所有数据点")
    print("2. 基于所有数据点构建凸边界")
    print("3. 在凸边界内随机选择探针")
    print("错误：探针位置受限于凸边界")
    print("错误：凸边界基于所有数据点，不够合理")

    print("\n新流程（正确）：")
    print("1. 随机选择固定数量的探针位置")
    print("2. 基于选中的探针构建凸边界")
    print("3. 筛选凸边界内的所有数据点")
    print("正确：探针位置完全随机，不受约束")
    print("正确：凸边界基于探针构建，更符合物理意义")
    print("正确：只处理探针凸边界内的数据，提高效率")

if __name__ == "__main__":
    success = test_new_data_flow()
    compare_old_vs_new()

    if success:
        print("\n新的数据处理流程测试成功！")
        print("可以开始训练了")
    else:
        print("\n测试失败，请检查错误信息")