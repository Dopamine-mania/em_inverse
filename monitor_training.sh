#!/bin/bash
# 训练监控脚本 - 明天早上查看结果用

echo "=========================================="
echo "  DeepONet 随机探针训练监控 (300 Epochs)"
echo "=========================================="
echo ""

# 检查训练进程
if ps aux | grep -q "[p]ython train.py"; then
    echo "✅ 训练进程运行中..."
    ps aux | grep "[p]ython train.py" | awk '{print "   PID:", $2, " | CPU:", $3"%", " | MEM:", $4"%"}'
else
    echo "⏹️  训练进程已结束"
fi

echo ""
echo "------------------------------------------"
echo "📊 训练进度"
echo "------------------------------------------"

# 提取最新的epoch信息
LATEST_EPOCH=$(grep -oP "Epoch \K\d+(?=/300)" training_random_probes_300ep.log 2>/dev/null | tail -1)
if [ -z "$LATEST_EPOCH" ]; then
    LATEST_EPOCH="未知"
fi
echo "当前Epoch: $LATEST_EPOCH / 300"

# 提取最新的Loss值
echo ""
echo "最新训练指标："
grep -E "\[Epoch.*Total:.*Pure MSE.*Test:" training_random_probes_300ep.log 2>/dev/null | tail -3

echo ""
echo "------------------------------------------"
echo "🏆 最佳模型记录"
echo "------------------------------------------"
grep "NEW BEST" training_random_probes_300ep.log 2>/dev/null | tail -5

echo ""
echo "------------------------------------------"
echo "📈 关键性能对比"
echo "------------------------------------------"

# 提取第1个epoch和最新epoch的Pure MSE
FIRST_PURE_MSE=$(grep -oP "Epoch 1/.*Pure MSE \(Data\): \K[\d\.]+" training_random_probes_300ep.log 2>/dev/null | head -1)
LATEST_PURE_MSE=$(grep -oP "Pure MSE \(Data\): \K[\d\.]+" training_random_probes_300ep.log 2>/dev/null | tail -1)

echo "Train Pure MSE:"
echo "  Epoch 1:      ${FIRST_PURE_MSE:-未知}"
echo "  Epoch Latest: ${LATEST_PURE_MSE:-未知}"

# 提取Test Loss变化
FIRST_TEST=$(grep -oP "Epoch 1/.*Test: \K[\d\.]+" training_random_probes_300ep.log 2>/dev/null | head -1)
LATEST_TEST=$(grep -oP "Test: \K[\d\.]+" training_random_probes_300ep.log 2>/dev/null | tail -1)

echo ""
echo "Test Loss:"
echo "  Epoch 1:      ${FIRST_TEST:-未知}"
echo "  Epoch Latest: ${LATEST_TEST:-未知}"

echo ""
echo "------------------------------------------"
echo "⏱️  训练时间统计"
echo "------------------------------------------"
if grep -q "总用时" training_random_probes_300ep.log 2>/dev/null; then
    grep "总用时" training_random_probes_300ep.log | tail -1
else
    echo "训练尚未完成..."
fi

echo ""
echo "=========================================="
echo "使用说明："
echo "  - 查看实时日志: tail -f training_random_probes_300ep.log"
echo "  - 查看Loss曲线: ls outputs/day2_fast_training/"
echo "  - 查看最佳模型: ls checkpoints/day2_fast_training/"
echo "=========================================="
