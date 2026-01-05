#!/bin/bash
# 自动化训练延长脚本 - 智能判断是否需要继续训练

LOG_300="training_random_probes_300ep.log"
LOG_1000="training_random_probes_1000ep.log"
CHECKPOINT_DIR="checkpoints/day2_fast_training"

echo "=========================================="
echo "  自动化训练延长系统"
echo "=========================================="
echo ""

# 等待300 epoch训练完成
echo "⏳ 等待300 epoch训练完成..."
while ps aux | grep -q "[p]ython train.py.*--epochs 300"; do
    sleep 30
    CURRENT_EPOCH=$(grep -oP "Epoch \K\d+(?=/300)" $LOG_300 2>/dev/null | tail -1)
    echo "   当前进度: Epoch $CURRENT_EPOCH / 300"
done

echo "✅ 300 epoch训练已完成！"
sleep 5

# 提取最后10个epoch的Test Loss
echo ""
echo "📊 分析最后10个epoch的Loss趋势..."
LAST_10_TEST=$(grep -oP "Test: \K[\d\.]+" $LOG_300 | tail -10)

# 计算最后5个epoch的平均值
LAST_5_AVG=$(echo "$LAST_10_TEST" | tail -5 | awk '{sum+=$1; count++} END {print sum/count}')
# 计算前5个epoch的平均值
FIRST_5_AVG=$(echo "$LAST_10_TEST" | head -5 | awk '{sum+=$1; count++} END {print sum/count}')

echo "   最后10个epoch中："
echo "   - 前5个平均Test Loss: $FIRST_5_AVG"
echo "   - 后5个平均Test Loss: $LAST_5_AVG"

# 计算下降百分比
IMPROVEMENT=$(echo "$FIRST_5_AVG $LAST_5_AVG" | awk '{print ($1-$2)/$1*100}')
echo "   - 改善幅度: $IMPROVEMENT%"

# 判断是否需要延长训练
THRESHOLD=1.0  # 如果最后5个epoch仍改善超过1%，则继续训练
SHOULD_CONTINUE=$(echo "$IMPROVEMENT $THRESHOLD" | awk '{if ($1 > $2) print "yes"; else print "no"}')

if [ "$SHOULD_CONTINUE" = "yes" ]; then
    echo ""
    echo "🚀 决策：Loss仍在下降（${IMPROVEMENT}% > ${THRESHOLD}%），启动1000 epoch训练！"
    echo ""
    
    # 清除300 epoch的checkpoint，避免冲突
    echo "🗑️  清理300 epoch的非最佳checkpoint..."
    find $CHECKPOINT_DIR -name "*.pth" ! -name "best_*.pth" -delete
    
    # 启动1000 epoch训练
    echo "🔥 启动1000 epoch长时间训练..."
    nohup python train.py --config config/day2_fast_training.yaml --epochs 1000 > $LOG_1000 2>&1 &
    NEW_PID=$!
    echo "   训练进程PID: $NEW_PID"
    echo "   日志文件: $LOG_1000"
    
    # 等待1000 epoch训练完成
    echo ""
    echo "⏳ 等待1000 epoch训练完成（预计20-30分钟）..."
    while ps -p $NEW_PID > /dev/null 2>&1; do
        sleep 60
        CURRENT_EPOCH=$(grep -oP "Epoch \K\d+(?=/1000)" $LOG_1000 2>/dev/null | tail -1)
        LATEST_TEST=$(grep -oP "Test: \K[\d\.]+" $LOG_1000 2>/dev/null | tail -1)
        echo "   当前进度: Epoch $CURRENT_EPOCH / 1000 | Test Loss: $LATEST_TEST"
    done
    
    echo "✅ 1000 epoch训练已完成！"
    FINAL_LOG=$LOG_1000
else
    echo ""
    echo "⏹️  决策：Loss已收敛（改善幅度${IMPROVEMENT}% < ${THRESHOLD}%），无需延长训练"
    echo "   使用300 epoch的模型作为最终结果"
    FINAL_LOG=$LOG_300
fi

echo ""
echo "=========================================="
echo "  开始自动生成可视化图表"
echo "=========================================="

# 调用可视化脚本
if [ -f "auto_visualize.py" ]; then
    echo "🎨 生成可视化图表..."
    python auto_visualize.py --log_file $FINAL_LOG
else
    echo "⚠️  可视化脚本未找到，请手动生成"
fi

echo ""
echo "=========================================="
echo "  训练完成总结"
echo "=========================================="
grep -E "最佳|BEST" $FINAL_LOG | tail -5

echo ""
echo "📁 输出文件位置："
echo "   - 最佳模型: $(ls -t $CHECKPOINT_DIR/best_*.pth 2>/dev/null | head -1)"
echo "   - 可视化图: outputs/day2_fast_training/"
echo "   - 完整日志: $FINAL_LOG"
echo ""
echo "✅ 全部自动化流程完成！明天早上直接查看结果即可。"

