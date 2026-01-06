#!/usr/bin/env python
"""
🔍 系统就绪验证脚本
==================

检查所有必要组件是否就绪，确保能够立即响应盲测。
"""

import sys
from pathlib import Path
import subprocess

def print_header(text):
    """打印标题"""
    print("\n" + "=" * 80)
    print(f"  {text}")
    print("=" * 80)

def check_python_version():
    """检查Python版本"""
    print("\n📋 检查Python版本...")
    version = sys.version_info
    if version.major == 3 and version.minor >= 11:
        print(f"   ✅ Python {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print(f"   ❌ Python版本过低: {version.major}.{version.minor}.{version.micro}")
        print(f"      需要 Python 3.11+")
        return False

def check_dependencies():
    """检查依赖包"""
    print("\n📦 检查依赖包...")
    packages = {
        'torch': 'PyTorch',
        'numpy': 'NumPy',
        'pandas': 'Pandas',
        'matplotlib': 'Matplotlib',
        'scipy': 'SciPy',
        'yaml': 'PyYAML'
    }

    all_ok = True
    for module, name in packages.items():
        try:
            __import__(module)
            print(f"   ✅ {name}")
        except ImportError:
            print(f"   ❌ {name} 未安装")
            all_ok = False

    return all_ok

def check_cuda():
    """检查CUDA可用性"""
    print("\n🎮 检查GPU/CUDA...")
    try:
        import torch
        if torch.cuda.is_available():
            print(f"   ✅ CUDA可用")
            print(f"   ✅ GPU数量: {torch.cuda.device_count()}")
            print(f"   ✅ GPU名称: {torch.cuda.get_device_name(0)}")
            return True
        else:
            print(f"   ⚠️  CUDA不可用，将使用CPU推理（速度较慢）")
            return True  # CPU也可以工作，只是慢一些
    except Exception as e:
        print(f"   ❌ 检查CUDA时出错: {e}")
        return False

def check_model_weights():
    """检查模型权重文件"""
    print("\n🧠 检查模型权重...")
    checkpoint_dir = Path('checkpoints/day2_fast_training')

    if not checkpoint_dir.exists():
        print(f"   ❌ 检查点目录不存在: {checkpoint_dir}")
        return False

    best_models = list(checkpoint_dir.glob('best_*.pth'))
    if not best_models:
        print(f"   ❌ 未找到最佳模型文件")
        return False

    best_model = max(best_models, key=lambda p: p.stat().st_mtime)
    size_mb = best_model.stat().st_size / (1024 * 1024)

    print(f"   ✅ 找到模型: {best_model.name}")
    print(f"   ✅ 文件大小: {size_mb:.2f} MB")

    if size_mb < 1:
        print(f"   ⚠️  警告: 模型文件可能不完整（小于1MB）")
        return False

    return True

def check_config_files():
    """检查配置文件"""
    print("\n⚙️  检查配置文件...")
    config_file = Path('config/day2_fast_training.yaml')

    if not config_file.exists():
        print(f"   ❌ 配置文件不存在: {config_file}")
        return False

    print(f"   ✅ 配置文件存在: {config_file}")
    return True

def check_inference_script():
    """检查推理脚本"""
    print("\n🔮 检查推理脚本...")
    script = Path('predict_new_data.py')

    if not script.exists():
        print(f"   ❌ 推理脚本不存在: {script}")
        return False

    print(f"   ✅ 推理脚本存在: {script}")

    # 检查脚本大小（确保不是空文件）
    size_kb = script.stat().st_size / 1024
    if size_kb < 1:
        print(f"   ❌ 推理脚本文件过小（可能为空）")
        return False

    print(f"   ✅ 脚本大小: {size_kb:.2f} KB")
    return True

def check_output_directory():
    """检查输出目录"""
    print("\n📁 检查输出目录...")
    output_dir = Path('outputs')

    if not output_dir.exists():
        print(f"   ⚠️  输出目录不存在，将自动创建")
        output_dir.mkdir(parents=True, exist_ok=True)

    print(f"   ✅ 输出目录就绪")
    return True

def check_delivery_package():
    """检查交付包"""
    print("\n📦 检查交付包...")
    delivery_dir = Path('Delivery_Package')

    if not delivery_dir.exists():
        print(f"   ⚠️  交付包目录不存在")
        return False

    required_files = [
        'README.md',
        'QUICK_START.md',
        'DEPLOYMENT_GUIDE.md',
        'requirements.txt',
        'predict_new_data.py'
    ]

    all_ok = True
    for file in required_files:
        file_path = delivery_dir / file
        if file_path.exists():
            print(f"   ✅ {file}")
        else:
            print(f"   ❌ 缺少: {file}")
            all_ok = False

    return all_ok

def run_quick_test():
    """运行快速测试"""
    print("\n🧪 运行快速测试...")
    try:
        from config.config import Config
        from model.enhanced_deeponet import SingleBranchDeepONet
        import torch

        print(f"   ✅ 配置加载器导入成功")
        print(f"   ✅ 模型定义导入成功")

        # 尝试加载配置
        cfg = Config(config_file='config/day2_fast_training.yaml')
        print(f"   ✅ 配置文件加载成功")

        # 尝试初始化模型
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = SingleBranchDeepONet(cfg).to(device)
        print(f"   ✅ 模型初始化成功")

        return True
    except Exception as e:
        print(f"   ❌ 测试失败: {e}")
        return False

def main():
    """主函数"""
    print_header("🔍 系统就绪验证 - 盲测准备检查")

    checks = [
        ("Python版本", check_python_version),
        ("依赖包", check_dependencies),
        ("GPU/CUDA", check_cuda),
        ("模型权重", check_model_weights),
        ("配置文件", check_config_files),
        ("推理脚本", check_inference_script),
        ("输出目录", check_output_directory),
        ("交付包", check_delivery_package),
        ("快速测试", run_quick_test)
    ]

    results = []
    for name, check_func in checks:
        try:
            result = check_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n   ❌ 检查 '{name}' 时出错: {e}")
            results.append((name, False))

    # 打印总结
    print_header("📊 检查结果汇总")
    all_passed = True
    for name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"   {status:12} | {name}")
        if not result:
            all_passed = False

    print("\n" + "=" * 80)
    if all_passed:
        print("🎉 恭喜！所有检查通过，系统已就绪！")
        print("=" * 80)
        print("\n⚡ 下一步：")
        print("   1. 等待客户发送盲测CSV文件")
        print("   2. 将CSV文件放入任意目录")
        print("   3. 运行: python predict_new_data.py --input_dir /path/to/csv")
        print("   4. 30分钟内查看 outputs/blind_test/ 目录获取结果")
        print("\n🚀 准备完毕，随时应对盲测挑战！")
        return 0
    else:
        print("❌ 部分检查未通过，请修复上述问题")
        print("=" * 80)
        print("\n💡 建议：")
        print("   1. 检查 requirements.txt 中的依赖是否完整安装")
        print("   2. 确认模型权重文件存在且完整")
        print("   3. 运行: pip install -r requirements.txt")
        return 1

if __name__ == "__main__":
    sys.exit(main())
