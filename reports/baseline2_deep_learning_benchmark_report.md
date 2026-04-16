# Baseline Model 2 深度学习模型对比报告

## 1. 实验设置
- 数据源：`Dataset2.xlsx`
- 训练/测试划分：8:2（分层抽样，随机种子=42）
- 训练集文件：`data\baseline2_train_split_80.csv`
- 测试集文件：`data\baseline2_test_split_20.csv`
- 说明：当前 `Dataset2.xlsx` 中未发现单独命名的 AI 标签列，本实验按现有标准化情感标签进行全量监督学习评估。

## 2. 模型对比结果
- **distilbert**：Accuracy=0.8878，Precision=0.8870，Recall=0.8878，F1=0.8874。
- **bert**：Accuracy=0.8847，Precision=0.8842，Recall=0.8847，F1=0.8844。
- **cnn**：Accuracy=0.8522，Precision=0.8736，Recall=0.8522，F1=0.8594。
- **lstm**：Accuracy=0.8323，Precision=0.8308，Recall=0.8323，F1=0.8309。

## 3. 推荐结论
- 当前 Baseline Model 2 最优模型：**distilbert**。
- 推荐依据：其加权 F1 最高（0.8874），可作为 Baseline Model 2 的主要汇报结果。
- 对比图：`visualizations\10_baseline2_model_comparison_bar.png`
- 混淆矩阵：`visualizations\12_baseline2_best_model_confusion_matrix.png`