# CDS529 Sentiment Analysis Pipeline

本项目用于课程 `CDS529`：基于客户评论数据完成基础与进阶情感分析，并输出主题级业务洞察。

## 1. 环境要求

- Python 3.8+
- 推荐在 Cursor 本地或 Google Colab 运行

安装依赖：

```bash
pip install -r requirements.txt
```

## 2. 输入数据

默认输入路径：

- `./data/cleaned_reviews_with_topics.csv`

若默认文件不存在，脚本会自动回退尝试：

- `./processed_reviews_dedup.csv`
- `./data/processed_reviews_dedup.csv`

## 3. 一键运行

```bash
python sentiment_analysis_pipeline.py
```

可选参数：

```bash
# 跳过 BERT 训练
python sentiment_analysis_pipeline.py --skip-bert-training

# 指定 BERT 训练轮数
python sentiment_analysis_pipeline.py --bert-epochs 3

# 仅运行 BERT 训练
python sentiment_analysis_pipeline.py --bert-only --bert-epochs 3

# 开启 Early Stopping（patience=2, threshold=0.0001）
python sentiment_analysis_pipeline.py --bert-only --bert-epochs 10 --bert-early-stopping-patience 2 --bert-early-stopping-threshold 0.0001

# 运行 Baseline Model 1：Dataset1.xlsx 8:2 划分 + 多模型深度学习对比
python sentiment_analysis_pipeline.py --run-baseline1 --baseline1-models distilbert,bert,albert --baseline1-epochs 2

# 运行 Baseline Model 1 预览版（小样本，便于快速看图）
python sentiment_analysis_pipeline.py --run-baseline1 --baseline1-models distilbert,bert --baseline1-epochs 1 --baseline1-preview-sample-size 600

# 加入传统深度学习模型进行纵向对比
python sentiment_analysis_pipeline.py --run-baseline1 --baseline1-models cnn,lstm,distilbert,bert --baseline1-epochs 2
```

## 4. 输出文件说明

### 数据与结果（`./data/`）

- `validated_reviews_for_sentiment.csv`：数据校验后的评论数据
- `baseline_vader_sentiment_results.csv`：VADER 情感分析结果
- `advanced_transformer_sentiment_results.csv`：Transformer 情感分析结果
- `validation_sample_50_for_manual_check.csv`：抽样50条用于人工复核
- `topic_sentiment_summary.csv`：主题级情感汇总（占比、平均分、NPS-like、负评关键词）
- `bert_eval_predictions.csv`：BERT 在验证集上的预测明细
- `baseline1_train_split_80.csv`：Dataset1.xlsx 的 80% 训练集
- `baseline1_test_split_20.csv`：Dataset1.xlsx 的 20% 测试集
- `baseline1_model_comparison_results.csv`：Baseline Model 1 多模型对比结果

### 报告（`./reports/`）

- `sentiment_model_validation_report.md`：模型验证报告
- `sentiment_analysis_business_insights.md`：业务洞察报告（中文）
- `bert_training_log.csv`：BERT 训练日志（loss / eval 指标）
- `bert_training_metrics.json`：BERT 最终评估指标与标签映射
- `baseline1_deep_learning_benchmark_report.md`：Baseline Model 1 对比实验报告

### 模型文件（`./models/`）

- `bert_sentiment_model/`：训练后的 BERT 模型与 tokenizer

### 图表（`./visualizations/`）

1. 整体情感分布饼图
2. 各主题情感分布分组柱状图
3. 主题平均情感得分排序柱状图
4. 痛点最高3个主题负评词云
5. VADER vs Transformer 模型对比图
6. BERT 训练曲线图（loss / metrics vs epoch）
7. Baseline Model 1 模型对比柱状图
8. Baseline Model 1 训练/测试标签分布图
9. Baseline Model 1 最优模型混淆矩阵

## 5. 代码结构（核心函数）

- `load_data`：加载与校验数据（缺失值、空文本、异常主题）
- `vader_sentiment`：VADER 基础情感分析
- `transformer_sentiment`：Transformer 进阶情感分析
- `train_bert_model`：BERT 监督训练与训练数据记录
- `run_baseline_model_1_experiment`：手工标注数据 8:2 划分与多模型深度学习对比
- `build_validation_report`：抽样验证与模型对比报告
- `compute_topic_summary`：按主题聚合统计与关键词提取
- `create_visualizations`：输出学术报告级图表
- `generate_business_report`：生成中文业务建议报告

## 6. 备注

- 结果可复现：已固定随机种子 `42`
- 如果样本中没有人工标签，脚本会生成人工标注抽样文件供后续复核
