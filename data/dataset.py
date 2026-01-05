"""
DeepONet电磁场数据加载器
实现固定探针数量的掩码DeepONet数据格式转换
"""

import numpy as np
import pandas as pd
import torch
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional, Generator
import random
from scipy.spatial import ConvexHull

from config import Config


class MaskedDeepONetDataset:
    """掩码DeepONet数据集类"""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.current_mask = None  # 存储当前batch的mask
        self.probe_count = cfg.data.num_probes  # 使用data配置中的探针数量
        self.fixed_probe_positions = cfg.data.fixed_probe_positions  # 是否固定探针位置
        self.fixed_probe_cache = {}  # 缓存固定的探针位置，按文件存储

        # 显示配置信息
        print(f"🔧 探针配置:")
        print(f"   探针数量: {self.probe_count}")
        print(f"   固定探针位置: {'启用' if self.fixed_probe_positions else '禁用'}")

    def _create_single_branch_input(self, probe_coords: np.ndarray,
                                     probe_real: np.ndarray,
                                     probe_imag: np.ndarray,
                                     frequency: float) -> np.ndarray:
        """
        生成单Branch交错格式输入: [x1, y1, real1, imag1, x2, y2, real2, imag2, ..., freq]

        Args:
            probe_coords: [num_probes, 2] 探针坐标 (x, y)
            probe_real: [num_probes] 实部测量值
            probe_imag: [num_probes] 虚部测量值
            frequency: 标量频率值

        Returns:
            branch_input: [num_probes * 4 + 1] 单Branch输入向量
        """
        branch_input = []
        for i in range(len(probe_coords)):
            branch_input.extend([
                probe_coords[i, 0],  # x坐标
                probe_coords[i, 1],  # y坐标
                probe_real[i],       # 实部
                probe_imag[i]        # 虚部
            ])
        branch_input.append(frequency)  # 频率放在最后
        return np.array(branch_input, dtype=np.float32)

    def peek_frequency(self, csv_file: str) -> Optional[float]:
        """轻量级频率检测，读取第一行的频率信息"""
        try:
            # 先读取第一行看是否有注释
            with open(csv_file, 'r') as f:
                first_line = f.readline().strip()

            if first_line.startswith('#'):
                # 跳过注释行，读取数据行
                df = pd.read_csv(csv_file, skiprows=1, nrows=1)
            else:
                # 直接读取数据行
                df = pd.read_csv(csv_file, nrows=1)

            # 查找频率列 (freq_1, freq_2, etc.)
            freq_columns = [col for col in df.columns if col.startswith('freq_')]

            if freq_columns:
                # 读取频率值
                freq_value = df[freq_columns[0]].iloc[0]
                return float(freq_value)
            else:
                print(f"警告：文件 {csv_file} 中未找到频率列，可用列: {df.columns.tolist()}")
                return None

        except Exception as e:
            print(f"警告：无法读取文件 {csv_file} 的频率信息: {e}")
            return None

    def peek_frequencies(self, csv_file: str) -> List[float]:
        """轻量级频率检测，读取文件中的所有频率信息"""
        try:
            # 先读取第一行看是否有注释
            with open(csv_file, 'r') as f:
                first_line = f.readline().strip()

            if first_line.startswith('#'):
                # 跳过注释行，读取数据行
                df = pd.read_csv(csv_file, skiprows=1, nrows=1)
            else:
                df = pd.read_csv(csv_file, nrows=1)

            # 查找所有频率列
            freq_columns = [col for col in df.columns if col.startswith('freq_')]
            if freq_columns:
                frequencies = []
                for freq_col in freq_columns:
                    freq_val = df[freq_col].iloc[0]
                    if pd.notna(freq_val):  # 确保频率值不是NaN
                        frequencies.append(float(freq_val))
                return frequencies
            else:
                print(f"警告：文件 {csv_file} 中未找到频率列，可用列: {df.columns.tolist()}")
                return []

        except Exception as e:
            print(f"警告：无法读取文件 {csv_file} 的频率信息: {e}")
            return []

    def scan_available_samples(self, data_path: str, max_frequency: float = 20.0,
                             max_samples: int = 800) -> List[Dict[str, Any]]:
        """扫描所有可用构型（真实构型，不预创建多个样本）
        Args:
            data_path: 数据文件路径
            max_frequency: 最大频率限制
            max_samples: 最大样本数
        """
        valid_samples = []
        data_path = Path(data_path)

        if not data_path.exists():
            print(f"数据路径不存在: {data_path}")
            return []

        csv_files = list(data_path.glob("*.csv"))
        print(f"发现 {len(csv_files)} 个CSV文件，正在提取构型...")

        for csv_file in csv_files:
            # 获取文件中的所有频率
            frequencies = self.peek_frequencies(str(csv_file))
            if frequencies:
                file_size = csv_file.stat().st_size

                # 为每个频率创建一个构型样本（不预创建多个探针选择）
                for freq_idx, frequency in enumerate(frequencies):
                    valid_samples.append({
                        'file': str(csv_file),
                        'frequency': frequency,
                        'freq_idx': freq_idx,
                        'size': file_size,
                        'source_file': csv_file.name
                    })

                    # 如果设置了最大样本数限制且已达到，则停止扫描
                    if max_samples is not None and len(valid_samples) >= max_samples:
                        print(f"达到最大样本数限制 ({max_samples})，停止扫描")
                        break
                else:
                    # 如果内层循环没有被break，继续处理下一个文件
                    continue
                # 如果内层循环被break，跳出外层循环
                break
            else:
                print(f"警告：文件 {csv_file.name} 中没有找到有效频率")

        print(f"总共发现 {len(valid_samples)} 个真实构型（不同频率的电磁场分布）")
        return valid_samples

    def prepare_single_sample(self, csv_file: str, freq_idx: int = 0, sample_idx: int = 0) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """将单个构型转换为DeepONet双分支格式：先选探针，再基于探针构建凸边界
        Args:
            csv_file: CSV文件路径
            freq_idx: 频率索引
            sample_idx: 样本索引，用于生成不同的探针选择
        Returns:
            branch_real_input: 实部探针输入 [探针数, 4] = (x, y, z, frequency)
            branch_imag_input: 虚部探针输入 [探针数, 4] = (x, y, z, frequency)
            trunk_input: 坐标输入 [内点数, 3] = (x, y, frequency)
            target_output: 目标场值 [内点数, 2] = (real, imag)
            probe_mask: 探针位置掩码 [内点数]
        """
        # 读取CSV文件，跳过注释行
        try:
            # 先读取第一行看是否有注释
            with open(csv_file, 'r') as f:
                first_line = f.readline().strip()

            if first_line.startswith('#'):
                df = pd.read_csv(csv_file, skiprows=1)
            else:
                df = pd.read_csv(csv_file)
        except Exception as e:
            print(f"读取CSV文件失败: {e}")
            raise ValueError(f"无法读取文件 {csv_file}")

        # 验证必需的列是否存在
        required_coordinate_columns = ['X', 'Y']
        missing_coord_columns = [col for col in required_coordinate_columns if col not in df.columns]
        if missing_coord_columns:
            raise ValueError(f"文件 {csv_file} 缺少必需的坐标列: {missing_coord_columns}. 可用列: {df.columns.tolist()}")

        # 查找频率和场值列
        columns = df.columns.tolist()
        freq_columns = [col for col in columns if col.startswith('freq_')]
        real_columns = [col for col in columns if col.startswith('Ez_real_')]
        imag_columns = [col for col in columns if col.startswith('Ez_imag_')]

        if not freq_columns:
            raise ValueError(f"文件 {csv_file} 中未找到频率列 (freq_1, freq_2等). 可用列: {df.columns.tolist()}")
        if not real_columns:
            raise ValueError(f"文件 {csv_file} 中未找到实部场值列 (Ez_real_1, Ez_real_2等). 可用列: {df.columns.tolist()}")
        if not imag_columns:
            raise ValueError(f"文件 {csv_file} 中未找到虚部场值列 (Ez_imag_1, Ez_imag_2等). 可用列: {df.columns.tolist()}")

        # 验证频率索引的有效性
        if freq_idx >= len(freq_columns):
            print(f"警告：频率索引 {freq_idx} 超出范围，使用第一个频率 (索引 0)")
            freq_idx = 0

        # 验证所有数组的长度是否一致
        if not (len(freq_columns) == len(real_columns) == len(imag_columns)):
            raise ValueError(f"文件 {csv_file} 中频率、实部、虚部列数量不一致: "
                           f"频率={len(freq_columns)}, 实部={len(real_columns)}, 虚部={len(imag_columns)}")

        # 提取坐标、频率和场值
        try:
            x = df['X'].values  # 使用正确的列名
            y = df['Y'].values

            # 调试：检查频率列
            selected_freq_col = freq_columns[freq_idx]
            selected_real_col = real_columns[freq_idx]
            selected_imag_col = imag_columns[freq_idx]

            # 验证数据类型和格式
            frequency_data = df[selected_freq_col]
            real_data = df[selected_real_col]
            imag_data = df[selected_imag_col]

            # 检查是否有NaN值
            if frequency_data.isna().all():
                raise ValueError(f"频率列 {selected_freq_col} 全部为NaN")
            if real_data.isna().all():
                raise ValueError(f"实部列 {selected_real_col} 全部为NaN")
            if imag_data.isna().all():
                raise ValueError(f"虚部列 {selected_imag_col} 全部为NaN")

            # 获取频率值（标量值）
            frequency = frequency_data.iloc[0]

            # 检查频率是否为有效数值
            if pd.isna(frequency):
                raise ValueError(f"文件 {csv_file} 中 {selected_freq_col} 的第一个值为NaN")

            try:
                frequency = float(frequency)
            except (ValueError, TypeError) as e:
                raise ValueError(f"频率值 '{frequency}' 无法转换为浮点数: {e}")

            # 获取场值数组
            real = real_data.values
            imag = imag_data.values

        except KeyError as e:
            print(f"提取字段失败，文件 {csv_file}: {e}")
            print(f"可用列名: {df.columns.tolist()}")
            print(f"频率列: {freq_columns}")
            print(f"实部列: {real_columns}")
            print(f"虚部列: {imag_columns}")
            raise ValueError(f"缺少必要的列: {e}")
        except ValueError as e:
            print(f"数据转换失败，文件 {csv_file}: {e}")
            raise e

        # 创建DataFrame
        df_processed = pd.DataFrame({
            'x': x,
            'y': y,
            'real': real,
            'imag': imag,
            'frequency': frequency
        })

        # 移除NaN值
        df_processed = df_processed.dropna()
        if len(df_processed) == 0:
            raise ValueError(f"文件 {csv_file} 清理后无有效数据")

        total_points = len(df_processed)

        # 步骤1：选择探针位置
        current_probe_count = self.probe_count
        if total_points < current_probe_count:
            # 静默处理探针数量调整
            current_probe_count = total_points

        if self.fixed_probe_positions:
            # 固定探针位置模式：所有样本使用相同的探针位置
            cache_key = f"{csv_file}_{freq_idx}"

            if cache_key not in self.fixed_probe_cache:
                # 第一次处理这个文件/频率组合，生成固定的探针位置
                print(f"🔒 生成固定探针位置: {Path(csv_file).name}, 频率{freq_idx}GHz")
                rng = np.random.RandomState(seed=hash(f"{csv_file}_{freq_idx}_FIXED_PROBES") % (2**32))
                probe_indices = rng.choice(total_points, current_probe_count, replace=False)
                self.fixed_probe_cache[cache_key] = probe_indices
                print(f"✅ 固定探针位置已缓存，探针数量: {len(probe_indices)}")
            else:
                # 使用缓存的固定探针位置
                probe_indices = self.fixed_probe_cache[cache_key]
                # 静默模式，不需要每次都打印
                if sample_idx == 0:  # 只在第一次打印，避免日志过多
                    print(f"📍 使用缓存的固定探针位置: {Path(csv_file).name}, 频率{freq_idx}GHz, 探针数: {len(probe_indices)}")
        else:
            # 随机探针位置模式：每个sample_idx使用不同的探针位置
            rng = np.random.RandomState(seed=hash(f"{csv_file}_{freq_idx}_{sample_idx}") % (2**32))
            probe_indices = rng.choice(total_points, current_probe_count, replace=False)
            if sample_idx == 0:  # 只在第一次打印，避免日志过多
                print(f"🎲 生成随机探针位置: {Path(csv_file).name}, 频率{freq_idx}GHz, sample_idx:{sample_idx}, 探针数: {len(probe_indices)}")

        # 步骤2：基于选中的探针构建凸边界
        all_coords = df_processed[['x', 'y']].values
        probe_coords = all_coords[probe_indices]  # 探针位置

        # 基于探针位置构建凸边界 - 改进版本：确保探针点始终包含在内
        try:
            hull = ConvexHull(probe_coords)
            hull_vertices = hull.vertices

            # 使用matplotlib.path来判断点是否在探针构成的凸边界内
            from matplotlib.path import Path as MatplotlibPath
            hull_path = MatplotlibPath(probe_coords[hull_vertices])

            # 关键修复1: 使用更大的radius确保边界点被包含
            interior_mask = hull_path.contains_points(all_coords, radius=1e-6)  # 增大radius

            # 关键修复2: 强制包含所有探针位置，确保探针不被凸包筛选掉
            # 创建最终的interior_mask：凸包内点 + 所有探针点
            probe_included_mask = np.zeros(len(all_coords), dtype=bool)
            for probe_idx in probe_indices:
                probe_included_mask[probe_idx] = True  # 强制包含所有探针

            # 合并：凸包内点 ∪ 探针点
            interior_mask = interior_mask | probe_included_mask

            # 步骤3：构建包含探针的完整数据集
            df_interior = df_processed[interior_mask].copy()
            interior_coords = all_coords[interior_mask]
            interior_fields = df_processed[['real', 'imag']].values[interior_mask]

            if len(df_interior) == 0:
                print(f"警告：构型 {Path(csv_file).name} 频率{freq_idx} 筛选后无数据点，使用所有数据")
                df_interior = df_processed.copy()
                interior_coords = all_coords
                interior_fields = df_processed[['real', 'imag']].values
                interior_mask = np.ones(len(all_coords), dtype=bool)

        except Exception as e:
            print(f"构型 {Path(csv_file).name} 频率{freq_idx} 探针凸边界计算失败: {e}，使用所有数据点")
            df_interior = df_processed.copy()
            interior_coords = all_coords
            interior_fields = df_processed[['real', 'imag']].values
            interior_mask = np.ones(len(all_coords), dtype=bool)

        # 关键修复3: 直接构建探针mask，避免复杂的坐标匹配
        # 由于我们强制包含了探针点，可以直接使用原始探针索引
        probe_mask = np.zeros(len(df_interior), dtype=bool)

        # 建立从原始索引到interior索引的映射
        # 这样可以直接找到探针在interior数据中的位置
        original_to_interior = {}
        interior_counter = 0

        for original_idx in range(len(all_coords)):
            if interior_mask[original_idx]:
                original_to_interior[original_idx] = interior_counter
                interior_counter += 1

        # 使用映射直接设置探针mask - 100%准确！
        for probe_idx in probe_indices:
            if probe_idx in original_to_interior:
                interior_idx = original_to_interior[probe_idx]
                probe_mask[interior_idx] = True
            else:
                print(f"错误：探针索引{probe_idx}未在映射中找到，这不应该发生")

        found_probes = np.sum(probe_mask)
        expected_probes = len(probe_indices)

        if found_probes != expected_probes:
            # 最后的安全检查：确保至少有探针位置
            if found_probes == 0:
                probe_mask[:min(expected_probes, len(probe_mask))] = True

        # 步骤4：构建单Branch训练数据 - 交错格式包含实部和虚部
        # 创建频率数组，并应用频率缩放（除以1GHz基准）
        if self.cfg.data.enable_frequency_scaling:
            frequency_scaled = frequency / self.cfg.data.frequency_scale_factor  # 将频率转换为以1GHz为基准的无量纲值
        else:
            frequency_scaled = frequency  # 使用原始频率值

        # probe_coords_only: 探针位置坐标 [探针数, 2] = (x, y)
        # 注意：我们电磁场数据是2D的，没有z坐标
        probe_coords_only = probe_coords  # [探针数, 2] (x, y)

        # 获取探针位置的测量场值
        probe_fields = df_processed[['real', 'imag']].values[probe_indices]  # [探针数, 2] (real, imag)
        probe_real_values = probe_fields[:, 0]  # [探针数] 实部测量值
        probe_imag_values = probe_fields[:, 1]  # [探针数] 虚部测量值

        # 生成单Branch输入 [探针数*4 + 1] = [x1,y1,r1,i1, x2,y2,r2,i2, ..., freq]
        branch_input = self._create_single_branch_input(
            probe_coords_only, probe_real_values, probe_imag_values, frequency_scaled
        )

        # trunk_input: 凸边界内所有位置坐标 [内点数, 3] = (x, y, frequency)
        # 注意：trunk网络使用2D坐标+频率，没有z坐标
        # 应用相同的频率缩放
        freq_array_interior = np.full((len(df_interior), 1), frequency_scaled)
        trunk_input = np.concatenate([interior_coords, freq_array_interior], axis=1)

        # target_output: 凸边界内所有场值 [内点数, 2] = (real, imag)
        target_output = interior_fields

        # probe_coords_3d: 探针坐标（扩展为3D，z=0）用于loss计算
        z_zeros = np.zeros((current_probe_count, 1))
        probe_coords_3d = np.concatenate([probe_coords_only, z_zeros], axis=1)

        # 返回: 单branch输入, trunk输入, 目标输出, 探针mask, 探针坐标
        return branch_input, trunk_input, target_output, probe_mask, probe_coords_3d

    def create_deeponet_dataset(self, samples_list: List[Dict[str, Any]],
                               train_ratio: float = 0.8) -> Tuple:
        """创建DeepONet格式数据集"""
        # 随机打乱样本
        np.random.shuffle(samples_list)

        # 划分训练集和测试集
        split_idx = int(len(samples_list) * train_ratio)
        train_samples = samples_list[:split_idx]
        test_samples = samples_list[split_idx:]

        # 准备训练数据
        return self._prepare_batch_data(train_samples, test_samples)

    def _prepare_batch_data(self, train_samples: List[Dict],
                           test_samples: List[Dict]) -> Tuple:
        """准备批次数据 - 处理可变长度数据（单Branch格式）"""
        # 准备训练数据
        branch_train_list = []  # 单Branch输入
        trunk_train_list = []
        y_train_list = []
        masks_train_list = []
        probe_coords_train_list = []  # 探针坐标

        print("准备训练数据...")
        for i, sample in enumerate(train_samples):
            try:
                # 使用样本中的频率索引
                freq_idx = sample.get('freq_idx', 0)
                branch_input, trunk_input, target_output, mask, probe_coords = self.prepare_single_sample(sample['file'], freq_idx=freq_idx)
                branch_train_list.append(branch_input)
                trunk_train_list.append(trunk_input)
                y_train_list.append(target_output)
                masks_train_list.append(mask)
                probe_coords_train_list.append(probe_coords)

                print(f"已处理 {i + 1}/{len(train_samples)} 个训练样本", end='\r')

            except ValueError as e:
                print(f"跳过训练样本 {sample.get('source_file', sample['file'])} (频率索引 {freq_idx}): {e}")
                # 跳过无效文件，继续处理其他文件
                continue
            except Exception as e:
                print(f"处理训练样本 {sample.get('source_file', sample['file'])} 时发生未知错误: {e}")
                # 跳过错误文件，继续处理
                continue

        # 完成后换行
        print()  # 换行
        print(f"训练数据准备完成: {len(branch_train_list)} 个有效样本")

        # 准备测试数据
        branch_test_list = []  # 单Branch输入
        trunk_test_list = []
        y_test_list = []
        masks_test_list = []
        probe_coords_test_list = []  # 探针坐标

        print("准备测试数据...")
        for i, sample in enumerate(test_samples):
            try:
                # 使用样本中的频率索引
                freq_idx = sample.get('freq_idx', 0)
                branch_input, trunk_input, target_output, mask, probe_coords = self.prepare_single_sample(sample['file'], freq_idx=freq_idx)
                branch_test_list.append(branch_input)
                trunk_test_list.append(trunk_input)
                y_test_list.append(target_output)
                masks_test_list.append(mask)
                probe_coords_test_list.append(probe_coords)

                print(f"已处理 {i + 1}/{len(test_samples)} 个测试样本", end='\r')

            except ValueError as e:
                print(f"跳过测试样本 {sample.get('source_file', sample['file'])} (频率索引 {freq_idx}): {e}")
                # 跳过无效文件，继续处理其他文件
                continue
            except Exception as e:
                print(f"处理测试样本 {sample.get('source_file', sample['file'])} 时发生未知错误: {e}")
                # 跳过错误文件，继续处理
                continue

        # 完成后换行
        print()  # 换行
        print(f"测试数据准备完成: {len(branch_test_list)} 个有效样本")

        # 不要直接合并为numpy数组，返回列表让训练器处理可变长度
        print(f"训练数据准备完成: {len(branch_train_list)} 个样本")
        print(f"测试数据准备完成: {len(branch_test_list)} 个样本")

        # 返回单Branch格式数据
        return (branch_train_list, trunk_train_list, y_train_list, masks_train_list, probe_coords_train_list,
                branch_test_list, trunk_test_list, y_test_list, masks_test_list, probe_coords_test_list)

    def get_data_statistics(self, samples_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """获取数据集统计信息"""
        if not samples_list:
            return {}

        frequencies = [sample['frequency'] for sample in samples_list]
        file_sizes = [sample['size'] for sample in samples_list]

        stats = {
            'total_samples': len(samples_list),
            'frequency_range': (min(frequencies) if frequencies else 0, max(frequencies) if frequencies else 0),
            'mean_frequency': np.mean(frequencies) if frequencies else 0,
            'std_frequency': np.std(frequencies) if frequencies else 0,
            'total_size_mb': sum(file_sizes) / (1024 * 1024),
            'avg_size_mb': np.mean(file_sizes) / (1024 * 1024) if file_sizes else 0
        }

        return stats


class OptimizedDataLoader:
    """内存优化的数据加载器"""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.batch_size = cfg.training.batch_size

    def load_data_generator(self, sample_files: List[Dict], mode: str = 'train') -> Generator[Dict, None, None]:
        """生成器模式加载数据，减少内存占用"""
        dataset = MaskedDeepONetDataset(self.cfg)

        while True:
            # 随机打乱
            np.random.shuffle(sample_files)

            # 分批处理
            for i in range(0, len(sample_files), self.batch_size):
                batch_files = sample_files[i:i + self.batch_size]

                branch_real_batch = []
                branch_imag_batch = []
                trunk_batch = []
                y_batch = []
                mask_batch = []

                for file_dict in batch_files:
                    try:
                        branch_real_input, branch_imag_input, trunk_input, target_output, mask = dataset.prepare_single_sample(file_dict['file'], freq_idx=0)
                        branch_real_batch.append(branch_real_input)
                        branch_imag_batch.append(branch_imag_input)
                        trunk_batch.append(trunk_input)
                        y_batch.append(target_output)
                        mask_batch.append(mask)
                    except Exception as e:
                        print(f"加载文件 {file_dict['file']} 时出错: {e}")
                        continue

                if branch_real_batch:  # 确保有有效数据
                    yield {
                        'branch_real': np.array(branch_real_batch),
                        'branch_imag': np.array(branch_imag_batch),
                        'trunk': np.array(trunk_batch),
                        'y': np.array(y_batch),
                        'masks': mask_batch
                    }

                # 清理临时变量
                del branch_real_batch, branch_imag_batch, trunk_batch, y_batch, mask_batch