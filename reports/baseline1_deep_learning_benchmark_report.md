# Baseline Model 1 深度学习模型对比报告

## 1. 实验设置
- 数据源：`Dataset1.xlsx`
- 训练/测试划分：8:2（分层抽样，随机种子=42）
- 训练集文件：`data\baseline1_train_split_80.csv`
- 测试集文件：`data\baseline1_test_split_20.csv`

## 2. 模型对比结果
- **roberta**：Accuracy=0.7960，Precision=0.7950，Recall=0.7960，F1=0.7915。
- **bert**：Accuracy=0.7870，Precision=0.7858，Recall=0.7870，F1=0.7844。
- **albert**：Accuracy=0.7762，Precision=0.7765，Recall=0.7762，F1=0.7737。
- **distilbert**：Accuracy=0.7762，Precision=0.7767，Recall=0.7762，F1=0.7713。
- **cnn**：Accuracy=0.7563，Precision=0.7638，Recall=0.7563，F1=0.7582。
- **lstm**：Accuracy=0.7310，Precision=0.7240，Recall=0.7310，F1=0.7263。

## 3. 推荐结论
- 当前 Baseline Model 1 最优模型：**roberta**。
- 推荐依据：其加权 F1 最高（0.7915），适合三分类情感任务汇报展示。
- 对比图：`visualizations\7_baseline1_model_comparison_bar.png`
- 混淆矩阵：`visualizations\9_baseline1_best_model_confusion_matrix.png`