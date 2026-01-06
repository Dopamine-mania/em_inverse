#!/bin/bash
# ========================================
# 示例使用脚本 - 电磁场逆问题推理
# ========================================

set -e  # 遇到错误立即退出

echo "============================================"
echo "🚀 电磁场逆问题推理 - 示例脚本"
echo "============================================"
echo ""

# ========================================
# 1. 环境检查
# ========================================
echo "📋 步骤 1/4: 检查环境..."

# 检查Python版本
python_version=$(python --version 2>&1 | awk '{print $2}')
echo "   ✓ Python版本: $python_version"

# 检查PyTorch
if python -c "import torch" 2>/dev/null; then
    torch_version=$(python -c "import torch; print(torch.__version__)")
    echo "   ✓ PyTorch版本: $torch_version"
else
    echo "   ✗ PyTorch未安装！"
    echo "   请运行: pip install -r requirements.txt"
    exit 1
fi

# 检查CUDA
cuda_available=$(python -c "import torch; print(torch.cuda.is_available())")
if [ "$cuda_available" = "True" ]; then
    echo "   ✓ CUDA可用"
else
    echo "   ⚠ CUDA不可用，将使用CPU推理（速度较慢）"
fi

echo ""

# ========================================
# 2. 数据准备
# ========================================
echo "📦 步骤 2/4: 准备数据..."

# 检查是否提供了数据目录
if [ -z "$1" ]; then
    echo "   ⚠ 未指定数据目录，使用默认目录: customer_data/"
    DATA_DIR="customer_data"
else
    DATA_DIR="$1"
    echo "   ✓ 数据目录: $DATA_DIR"
fi

# 检查数据目录是否存在
if [ ! -d "$DATA_DIR" ]; then
    echo "   ✗ 错误: 数据目录不存在: $DATA_DIR"
    echo ""
    echo "用法: $0 [数据目录路径]"
    echo "示例: $0 /path/to/customer/csv/files"
    exit 1
fi

# 统计CSV文件数量
csv_count=$(find "$DATA_DIR" -name "*.csv" -type f | wc -l)
echo "   ✓ 找到 $csv_count 个CSV文件"

if [ "$csv_count" -eq 0 ]; then
    echo "   ✗ 错误: 数据目录中没有CSV文件"
    exit 1
fi

echo ""

# ========================================
# 3. 运行推理
# ========================================
echo "🔮 步骤 3/4: 运行推理..."
echo ""

# 记录开始时间
start_time=$(date +%s)

# 运行推理脚本
python predict_new_data.py \
    --input_dir "$DATA_DIR" \
    --output_dir outputs/blind_test

# 记录结束时间
end_time=$(date +%s)
duration=$((end_time - start_time))

echo ""

# ========================================
# 4. 结果汇总
# ========================================
echo "📊 步骤 4/4: 结果汇总..."
echo ""

# 检查输出目录
if [ -d "outputs/blind_test" ]; then
    echo "   ✓ 输出目录: outputs/blind_test/"

    # 统计生成的文件
    png_count=$(find outputs/blind_test -name "*.png" -type f | wc -l)
    echo "   ✓ 生成可视化图: $png_count 张"

    # 显示报告
    if [ -f "outputs/blind_test/inference_report.txt" ]; then
        echo "   ✓ 推理报告已生成"
        echo ""
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo "📄 推理报告摘要:"
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

        # 提取关键信息
        grep -A 5 "统计汇总" outputs/blind_test/inference_report.txt || true

        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    fi
else
    echo "   ✗ 错误: 输出目录未生成"
    exit 1
fi

echo ""
echo "============================================"
echo "✅ 推理完成！"
echo "============================================"
echo "⏱️  总耗时: ${duration} 秒"
echo ""
echo "📁 结果位置:"
echo "   - 可视化图: outputs/blind_test/*.png"
echo "   - 详细报告: outputs/blind_test/inference_report.txt"
echo ""
echo "🔍 查看完整报告:"
echo "   cat outputs/blind_test/inference_report.txt"
echo ""
echo "🖼️  查看可视化:"
echo "   ls outputs/blind_test/*.png"
echo "============================================"

# ========================================
# 性能提示
# ========================================
if [ "$duration" -gt 1800 ]; then
    echo ""
    echo "⚠️  注意: 推理耗时超过30分钟 (${duration}秒)"
    echo "   建议检查:"
    echo "   - GPU是否可用"
    echo "   - 数据文件数量是否过多"
fi

exit 0
