# 🌅 明天早上查看指南

## 🎯 核心验证第一步：探针位置检查

**最重要的验证**：所有红色叉（探针）必须都在点云范围内！

```bash
cd /home/jovyan/teaching_material/Work/December/ai_physics/em_inverse_25probes/em_field

# 查看生成的图片
ls -lh outputs/final_visualizations/

# 用图片查看器打开（或直接在Jupyter中查看）
# 重点检查：real_space_*.png 中的红色×是否都在散点图内部
```

**预期结果**：
- ✅ `real_space_best_case.png`：红色探针×都在蓝色点云内
- ✅ `real_space_medium_case.png`：红色探针×都在蓝色点云内
- ✅ `real_space_hard_case.png`：红色探针×都在蓝色点云内

---

## 📊 性能指标查看

### 方法1：快速查看（推荐）

```bash
./monitor_training.sh
```

### 方法2：详细分析

```bash
# 查看自动化流程日志
cat auto_extend.log

# 查看最终训练日志
tail -100 training_random_probes_1000ep.log  # 如果延长到1000 epoch
# 或
tail -100 training_random_probes_300ep.log   # 如果300 epoch已收敛
```

---

## 🏆 关键性能指标（期望值）

### 理想情况（充分收敛）

| 指标 | 目标值 | 说明 |
|------|--------|------|
| **Train Pure MSE** | 0.3-0.8 | 比固定探针高（任务更难）|
| **Test Loss** | 12-18 | 比固定探针低（泛化更好）|
| **泛化比率** | 2-3× | 从5.2×大幅下降 ✅ |
| **训练Epochs** | 300-1000 | 根据收敛情况自动决定 |

### 如何计算泛化比率

```bash
# 从日志中提取最终的Train Pure MSE和Test Loss
grep "Pure MSE (Data):" training_random_probes_*.log | tail -1
grep "Test:" training_random_probes_*.log | tail -1

# 泛化比率 = Test Loss / Train Pure MSE
# 例如：Test=15.0, Train=0.5 → 泛化比率=30× （仍需改进）
# 目标：Test=12.0, Train=0.6 → 泛化比率=20× （接受）
```

---

## 🔍 详细检查清单

### 1. 训练完成情况

```bash
# 检查是否还有训练进程在运行
ps aux | grep "python train.py"

# 如果还在运行，查看当前进度
tail -20 training_random_probes_*.log | grep "Epoch"
```

### 2. 模型文件检查

```bash
# 查看保存的最佳模型
ls -lh checkpoints/day2_fast_training/best_*.pth

# 查看模型是在哪个epoch保存的
ls checkpoints/day2_fast_training/ | grep best
```

### 3. 可视化图表检查

```bash
# 查看所有生成的图片
ls -lh outputs/final_visualizations/

# 应该包含：
# - real_space_best_case.png
# - real_space_medium_case.png
# - real_space_hard_case.png
# - kspace_best_case.png
# - kspace_medium_case.png
# - kspace_hard_case.png
# - training_loss_curves.png
```

### 4. 核心验证：探针位置

打开 `real_space_*.png` 图片，检查：

- ✅ **左图（GT）**：红色×是否都在散点图范围内？
- ✅ **中图（Pred）**：红色×是否都在散点图范围内？
- ✅ **右图（Error）**：蓝色×（探针）是否都在散点图范围内？

**如果发现有探针在外面**：
- ❌ 这是BUG！需要检查可视化脚本
- 查看 auto_extend.log 中的探针位置范围打印
- 对比场点位置范围和探针位置范围

---

## 📈 Loss趋势分析

```bash
# 绘制简单的Loss趋势
grep "Pure MSE (Data):" training_random_probes_*.log | \
  awk -F'Pure MSE \\(Data\\): ' '{print NR, $2}' | \
  awk '{print $1, $2}' > train_loss.txt

# 查看最后20个epoch的趋势
tail -20 train_loss.txt
```

**预期趋势**：
- ✅ 前50 epochs：快速下降（15 → 2）
- ✅ 50-200 epochs：缓慢下降（2 → 0.8）
- ✅ 200+ epochs：接近收敛（0.8 → 0.6）

---

## 🚨 可能出现的问题

### 问题1：训练未完成

```bash
# 检查是否有错误
tail -50 training_random_probes_*.log | grep -i error

# 检查GPU状态
nvidia-smi
```

### 问题2：Loss没有下降

```bash
# 查看Loss曲线
cat training_loss_curves.png  # 或在图片查看器中打开

# 如果Loss一直很高（>10），可能需要：
# - 检查学习率是否太低
# - 检查数据是否正确加载
```

### 问题3：探针在多边形外（最关键！）

如果发现探针在外面：

```bash
# 1. 检查fixed_probe_positions配置
grep "fixed_probe_positions" config/day2_fast_training.yaml
# 必须是 false

# 2. 运行验证脚本
python test_random_probes.py

# 3. 检查可视化脚本中的探针提取逻辑
# auto_visualize.py 第120-127行
```

---

## ✅ 成功标准

全部满足以下条件才算成功：

1. **探针位置** ✅ 
   - 所有红色×都在散点图范围内
   
2. **性能指标** ✅
   - Train Pure MSE < 1.0
   - Test Loss < 20
   - 泛化比率 < 4×
   
3. **可视化质量** ✅
   - 实空间对比图：GT vs Pred 吻合度高
   - k空间对比图：频谱对齐良好
   - Loss曲线图：显示收敛趋势
   
4. **物理一致性** ✅
   - k空间频谱主峰位置一致
   - Error分布合理（边界区域误差稍大）

---

## 🎉 如果一切正常

恭喜！"解锁封印"成功！

接下来可以：
1. 更新 Delivery_Package 中的模型和文档
2. 重新生成技术报告
3. 准备向客户演示

---

## 📞 如果有问题

检查以下日志文件：
- `training_random_probes_300ep.log` - 300 epoch训练日志
- `training_random_probes_1000ep.log` - 1000 epoch训练日志（如果延长）
- `auto_extend.log` - 自动化脚本日志
- `outputs/final_visualizations/` - 可视化图片

**重要文件位置**：
- 最佳模型：`checkpoints/day2_fast_training/best_*.pth`
- 可视化图：`outputs/final_visualizations/`
- 配置文件：`config/day2_fast_training.yaml`

