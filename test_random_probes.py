"""
验证随机探针位置是否正确工作
"""
import sys
import numpy as np
from config.config import Config
from data.dataset import MaskedDeepONetDataset

def test_random_probe_positions():
    """测试不同文件的探针位置是否真的不同"""

    # 加载配置
    cfg = Config(config_file='config/day2_fast_training.yaml')

    print(f"配置加载成功！")
    print(f"  fixed_probe_positions: {cfg.data.fixed_probe_positions}")
    print(f"  num_probes: {cfg.data.num_probes}")

    # 创建数据集
    dataset = MaskedDeepONetDataset(cfg)

    # 测试3个不同的文件
    test_files = [
        '/home/jovyan/teaching_material/Work/December/ai_physics/demands/01_电磁场数据文件/100.csv',
        '/home/jovyan/teaching_material/Work/December/ai_physics/demands/01_电磁场数据文件/101.csv',
        '/home/jovyan/teaching_material/Work/December/ai_physics/demands/01_电磁场数据文件/102.csv'
    ]

    print("\n" + "="*60)
    print("测试1：验证不同文件的探针坐标是否不同")
    print("="*60)

    probe_coords_list = []

    for csv_file in test_files:
        print(f"\n处理文件: {csv_file.split('/')[-1]}")

        # 准备单个样本
        branch_input, trunk_input, target_output, mask, probe_coords = dataset.prepare_single_sample(
            csv_file, freq_idx=0, sample_idx=0
        )

        # Branch输入格式: [x1, y1, real1, imag1, ..., x25, y25, real25, imag25, freq]
        # 提取前5个探针的坐标
        print(f"  Branch input维度: {branch_input.shape}")
        print(f"  前5个探针的坐标：")
        for i in range(5):
            x = branch_input[i*4]
            y = branch_input[i*4+1]
            real = branch_input[i*4+2]
            imag = branch_input[i*4+3]
            print(f"    探针{i+1}: x={x:.6f}, y={y:.6f}, real={real:.6f}, imag={imag:.6f}")

        probe_coords_list.append(branch_input[:20].copy())  # 保存前5个探针的(x,y,r,i)

    # 验证探针坐标是否不同
    print("\n" + "="*60)
    print("验证结果：")
    print("="*60)

    same_count = 0
    for i in range(len(probe_coords_list)):
        for j in range(i+1, len(probe_coords_list)):
            if np.allclose(probe_coords_list[i], probe_coords_list[j], atol=1e-6):
                same_count += 1
                print(f"⚠️ 文件 {test_files[i].split('/')[-1]} 和 {test_files[j].split('/')[-1]} 的探针坐标相同！")

    if same_count == 0:
        print("✅ 验证通过：所有文件的探针位置都不相同！")
        print("✅ fixed_probe_positions=false 配置生效！")
        return True
    else:
        print(f"❌ 验证失败：有 {same_count} 对文件的探针位置相同！")
        print("❌ fixed_probe_positions 配置可能未生效！")
        return False

if __name__ == "__main__":
    success = test_random_probe_positions()
    sys.exit(0 if success else 1)
