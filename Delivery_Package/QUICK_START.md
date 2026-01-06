# ⚡ 快速开始指南

## 🎯 30秒快速上手

```bash
# 1. 激活环境
conda activate em_inference

# 2. 准备数据
mkdir -p customer_data
cp /path/to/customer/*.csv customer_data/

# 3. 运行推理
python predict_new_data.py --input_dir customer_data

# 4. 查看结果
cat outputs/blind_test/inference_report.txt
```

---

## 📊 预期输出

```
================================================================================
🎉 推理完成！
================================================================================
✅ 处理文件数: 5
✅ 总样本数: 10
✅ 平均 MSE: 0.05234567
✅ 平均 Max Error: 1.23456789
⏱️  总耗时: 45.67 秒
📁 输出目录: outputs/blind_test
📄 详细报告: outputs/blind_test/inference_report.txt
================================================================================
```

---

## 🖼️ 可视化结果

### 实空间图示例

```
[Ground Truth]  [Prediction]  [Error]
     •••             •••         •••
   • × × •         • × × •     • × × •
  •  ×××  •       •  ×××  •   •  ×××  •
 •   ×××   •     •   ×××   • •   ×××   •
```

**关键特征**:
- ✅ 25个红色 × 探针标记清晰可见
- ✅ GT和Pred视觉高度相似
- ✅ Error图误差集中在低值区域

---

## 🆘 遇到问题？

### GPU不可用
```bash
export CUDA_VISIBLE_DEVICES=""
python predict_new_data.py --input_dir customer_data
```

### 内存不足
```bash
# 一次处理一个文件
for f in customer_data/*.csv; do
    python predict_new_data.py --input_dir $(dirname $f)
done
```

### 详细帮助
```bash
python predict_new_data.py --help
```

---

完整文档请参考: `DEPLOYMENT_GUIDE.md`
