"""
CDS529 情感分析全流程脚本

功能覆盖：
1) 数据加载与校验
2) VADER 基础情感分析
3) Transformer 进阶情感分析
4) 模型验证（50条样本）
5) 按主题聚合统计
6) 可视化产出
7) 业务洞察报告生成

运行方式：
python sentiment_analysis_pipeline.py
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import traceback
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from wordcloud import WordCloud


# ---------------------------
# 全局配置
# ---------------------------
RANDOM_SEED = 42
DEFAULT_INPUT_PATH = Path("./data/cleaned_reviews_with_topics.csv")
FALLBACK_INPUT_PATHS = [
    Path("./processed_reviews_dedup.csv"),
    Path("./data/processed_reviews_dedup.csv"),
]

DATA_DIR = Path("./data")
REPORT_DIR = Path("./reports")
VIZ_DIR = Path("./visualizations")
MODEL_DIR = Path("./models")

VALIDATED_PATH = DATA_DIR / "validated_reviews_for_sentiment.csv"
VADER_RESULT_PATH = DATA_DIR / "baseline_vader_sentiment_results.csv"
TRANSFORMER_RESULT_PATH = DATA_DIR / "advanced_transformer_sentiment_results.csv"
TOPIC_SUMMARY_PATH = DATA_DIR / "topic_sentiment_summary.csv"
VALIDATION_SAMPLE_PATH = DATA_DIR / "validation_sample_50_for_manual_check.csv"
VALIDATION_REPORT_PATH = REPORT_DIR / "sentiment_model_validation_report.md"
BUSINESS_REPORT_PATH = REPORT_DIR / "sentiment_analysis_business_insights.md"
BERT_MODEL_OUTPUT_DIR = MODEL_DIR / "bert_sentiment_model"
BERT_METRICS_PATH = REPORT_DIR / "bert_training_metrics.json"
BERT_TRAIN_LOG_PATH = REPORT_DIR / "bert_training_log.csv"
BERT_EVAL_PREDICTIONS_PATH = DATA_DIR / "bert_eval_predictions.csv"
BERT_TRAIN_CURVE_PATH = VIZ_DIR / "6_bert_training_curves.png"

BASELINE1_INPUT_PATH = Path("./Dataset1.xlsx")
BASELINE1_CLEAN_PATH = DATA_DIR / "baseline1_manual_labeled_cleaned.csv"
BASELINE1_TRAIN_PATH = DATA_DIR / "baseline1_train_split_80.csv"
BASELINE1_TEST_PATH = DATA_DIR / "baseline1_test_split_20.csv"
BASELINE1_RESULTS_PATH = DATA_DIR / "baseline1_model_comparison_results.csv"
BASELINE1_REPORT_PATH = REPORT_DIR / "baseline1_deep_learning_benchmark_report.md"
BASELINE1_VIZ_COMPARISON_PATH = VIZ_DIR / "7_baseline1_model_comparison_bar.png"
BASELINE1_VIZ_SPLIT_PATH = VIZ_DIR / "8_baseline1_train_test_distribution.png"
BASELINE1_VIZ_CONFUSION_PATH = VIZ_DIR / "9_baseline1_best_model_confusion_matrix.png"

BASELINE2_INPUT_PATH = Path("./Dataset2.xlsx")
BASELINE2_CLEAN_PATH = DATA_DIR / "baseline2_manual_ai_labeled_cleaned.csv"
BASELINE2_TRAIN_PATH = DATA_DIR / "baseline2_train_split_80.csv"
BASELINE2_TEST_PATH = DATA_DIR / "baseline2_test_split_20.csv"
BASELINE2_RESULTS_PATH = DATA_DIR / "baseline2_model_comparison_results.csv"
BASELINE2_REPORT_PATH = REPORT_DIR / "baseline2_deep_learning_benchmark_report.md"
BASELINE2_VIZ_COMPARISON_PATH = VIZ_DIR / "10_baseline2_model_comparison_bar.png"
BASELINE2_VIZ_SPLIT_PATH = VIZ_DIR / "11_baseline2_train_test_distribution.png"
BASELINE2_VIZ_CONFUSION_PATH = VIZ_DIR / "12_baseline2_best_model_confusion_matrix.png"

BASELINE1_MODEL_REGISTRY = {
    "bert": "bert-base-uncased",
    "distilbert": "distilbert-base-uncased",
    "albert": "albert-base-v2",
    "roberta": "roberta-base",
    "cnn": "TextCNN",
    "lstm": "BiLSTM",
}

TRADITIONAL_DEEP_LEARNING_MODELS = {"cnn", "lstm"}
TRADITIONAL_PAD_TOKEN = "<PAD>"
TRADITIONAL_UNK_TOKEN = "<UNK>"


def set_random_seed(seed: int = RANDOM_SEED) -> None:
    """设置随机种子，保证可复现。"""
    random.seed(seed)
    np.random.seed(seed)


def ensure_directories() -> None:
    """确保输出目录存在。"""
    for directory in [DATA_DIR, REPORT_DIR, VIZ_DIR, MODEL_DIR]:
        directory.mkdir(parents=True, exist_ok=True)


def detect_input_path(input_path: Path) -> Path:
    """
    检测数据文件路径。
    - 优先使用用户指定默认路径
    - 若不存在，尝试备用路径
    """
    if input_path.exists():
        return input_path

    for fallback in FALLBACK_INPUT_PATHS:
        if fallback.exists():
            print(f"[INFO] 默认路径不存在，已回退至: {fallback}")
            return fallback

    raise FileNotFoundError(
        "未找到输入数据文件。请确认存在以下任一路径："
        f"{[str(DEFAULT_INPUT_PATH)] + [str(p) for p in FALLBACK_INPUT_PATHS]}"
    )


def choose_column(columns: List[str], candidates: List[str]) -> Optional[str]:
    """在候选字段中选择第一个存在于数据集中的字段名。"""
    column_set = set(columns)
    for col in candidates:
        if col in column_set:
            return col
    return None


def normalize_sentiment_label(value: object) -> Optional[str]:
    """
    将多种人工标签格式映射到统一标签：
    positive / negative / neutral
    """
    if pd.isna(value):
        return None

    text = str(value).strip().lower()
    mapping = {
        "positive": "positive",
        "pos": "positive",
        "1": "positive",
        "negative": "negative",
        "neg": "negative",
        "-1": "negative",
        "0": "neutral",
        "neutral": "neutral",
        "neu": "neutral",
    }
    return mapping.get(text, None)


def load_data(
    input_path: Path = DEFAULT_INPUT_PATH,
) -> Tuple[pd.DataFrame, str, str, Optional[str]]:
    """
    加载并校验数据，返回：
    - 校验后的 DataFrame
    - 文本字段名
    - 主题字段名
    - 人工标签字段名（可选）
    """
    path = detect_input_path(input_path)
    try:
        df = pd.read_csv(path)
    except Exception as exc:
        raise RuntimeError(f"读取数据失败: {path}，错误: {exc}") from exc

    if df.empty:
        raise ValueError("输入数据为空，无法继续分析。")

    text_col = choose_column(
        df.columns.tolist(),
        ["text_clean", "clean_text", "processed_text", "text", "review", "comment"],
    )
    topic_col = choose_column(
        df.columns.tolist(),
        ["topic", "topic_label", "topic_name", "bertopic_topic", "lda_topic", "category"],
    )
    manual_label_col = choose_column(
        df.columns.tolist(),
        ["manual_label", "human_label", "sentiment_label", "label"],
    )

    if text_col is None:
        raise KeyError(
            "未找到文本字段。请确保至少包含以下字段之一："
            "text_clean/clean_text/processed_text/text/review/comment"
        )
    if topic_col is None:
        raise KeyError(
            "未找到主题字段。请确保至少包含以下字段之一："
            "topic/topic_label/topic_name/bertopic_topic/lda_topic/category"
        )

    # 统一类型，处理缺失与空文本
    df[text_col] = df[text_col].fillna("").astype(str)
    before_rows = len(df)
    df = df[df[text_col].str.strip() != ""].copy()
    dropped_rows = before_rows - len(df)

    # 主题字段校验与清洗
    df[topic_col] = df[topic_col].fillna("Unknown").astype(str).str.strip()
    df.loc[df[topic_col] == "", topic_col] = "Unknown"

    if manual_label_col is not None:
        df["manual_label_normalized"] = df[manual_label_col].apply(normalize_sentiment_label)
    else:
        df["manual_label_normalized"] = None

    # 输出基础统计信息
    print("\n========== 数据集基础信息 ==========")
    print(f"数据文件: {path}")
    print(f"总评论条数(清洗后): {len(df)}")
    print(f"主题数量: {df[topic_col].nunique()}")
    print(f"删除空文本条数: {dropped_rows}")
    print(f"文本字段: {text_col}")
    print(f"主题字段: {topic_col}")
    print(f"人工标签字段: {manual_label_col if manual_label_col else '无'}")
    print("\n样本预览:")
    print(df[[text_col, topic_col]].head(5))

    df.to_csv(VALIDATED_PATH, index=False, encoding="utf-8-sig")
    print(f"\n[OK] 已保存校验数据: {VALIDATED_PATH}")

    return df, text_col, topic_col, manual_label_col


def safe_import_vader():
    """安全导入并初始化 VADER 模型，自动补下载词典。"""
    try:
        import nltk
        from nltk.sentiment import SentimentIntensityAnalyzer

        try:
            _ = SentimentIntensityAnalyzer()
        except LookupError:
            nltk.download("vader_lexicon", quiet=True)
        return SentimentIntensityAnalyzer()
    except Exception as exc:
        raise RuntimeError(f"VADER 初始化失败: {exc}") from exc


def classify_vader_compound(compound: float) -> str:
    """根据 VADER compound 分数映射为三分类情感标签。"""
    if compound >= 0.05:
        return "positive"
    if compound <= -0.05:
        return "negative"
    return "neutral"


def vader_sentiment(df: pd.DataFrame, text_col: str) -> pd.DataFrame:
    """执行 VADER 情感分析并返回结果 DataFrame。"""
    analyzer = safe_import_vader()
    result_df = df.copy()

    vader_pos = []
    vader_neu = []
    vader_neg = []
    vader_compound = []
    vader_label = []

    for text in result_df[text_col].tolist():
        try:
            scores = analyzer.polarity_scores(text if isinstance(text, str) else "")
            compound = float(scores.get("compound", 0.0))
            pos = float(scores.get("pos", 0.0))
            neu = float(scores.get("neu", 0.0))
            neg = float(scores.get("neg", 0.0))
        except Exception:
            compound, pos, neu, neg = 0.0, 0.0, 1.0, 0.0

        vader_compound.append(compound)
        vader_pos.append(pos)
        vader_neu.append(neu)
        vader_neg.append(neg)
        vader_label.append(classify_vader_compound(compound))

    result_df["vader_label"] = vader_label
    result_df["vader_compound"] = vader_compound
    result_df["vader_pos_conf"] = vader_pos
    result_df["vader_neu_conf"] = vader_neu
    result_df["vader_neg_conf"] = vader_neg

    result_df.to_csv(VADER_RESULT_PATH, index=False, encoding="utf-8-sig")
    print(f"[OK] VADER 结果已保存: {VADER_RESULT_PATH}")
    return result_df


def safe_transformer_pipeline(model_name: str):
    """安全加载 Transformer pipeline。"""
    try:
        from transformers import pipeline
    except Exception as exc:
        raise RuntimeError(
            "无法导入 transformers。请先安装依赖: pip install transformers torch"
        ) from exc

    try:
        clf = pipeline(
            "sentiment-analysis",
            model=model_name,
            tokenizer=model_name,
            truncation=True,
            max_length=256,
        )
        return clf
    except Exception as exc:
        raise RuntimeError(f"Transformer 模型加载失败: {exc}") from exc


def map_transformer_output(raw_label: str, score: float) -> Tuple[str, float]:
    """
    将模型输出映射到统一标签和带方向分数。
    - 统一标签：positive / negative
    - 分数范围：[-1, 1]
    """
    label = str(raw_label).strip().upper()
    confidence = float(score)

    if "NEG" in label:
        return "negative", -confidence
    return "positive", confidence


def transformer_sentiment(
    df: pd.DataFrame,
    text_col: str,
    model_name: str = "distilbert-base-uncased-finetuned-sst-2-english",
    batch_size: int = 32,
) -> pd.DataFrame:
    """执行 Transformer 情感分析并返回结果 DataFrame。"""
    clf = safe_transformer_pipeline(model_name=model_name)
    result_df = df.copy()

    labels: List[str] = []
    confidences: List[float] = []
    scores: List[float] = []

    texts = result_df[text_col].fillna("").astype(str).tolist()

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        try:
            batch_output = clf(batch_texts)
        except Exception:
            # 批失败时逐条兜底，避免整个流程中断
            batch_output = []
            for t in batch_texts:
                try:
                    single = clf(t)[0]
                except Exception:
                    single = {"label": "NEUTRAL", "score": 0.0}
                batch_output.append(single)

        for out in batch_output:
            raw_label = out.get("label", "NEUTRAL")
            raw_score = float(out.get("score", 0.0))
            mapped_label, signed_score = map_transformer_output(raw_label, raw_score)
            labels.append(mapped_label)
            confidences.append(raw_score)
            scores.append(signed_score)

    result_df["transformer_label"] = labels
    result_df["transformer_confidence"] = confidences
    result_df["transformer_score"] = scores

    result_df.to_csv(TRANSFORMER_RESULT_PATH, index=False, encoding="utf-8-sig")
    print(f"[OK] Transformer 结果已保存: {TRANSFORMER_RESULT_PATH}")
    return result_df


def calculate_accuracy(y_true: pd.Series, y_pred: pd.Series) -> Optional[float]:
    """计算准确率；若有效样本为空则返回 None。"""
    valid_mask = y_true.notna() & y_pred.notna()
    if valid_mask.sum() == 0:
        return None
    return float((y_true[valid_mask] == y_pred[valid_mask]).mean())


def build_validation_report(df: pd.DataFrame) -> Dict[str, Optional[float]]:
    """
    抽取 50 条样本并生成模型验证报告。
    说明：
    - 若存在人工标签字段，则直接计算准确率
    - 若不存在，则输出人工标注模板，准确率留空
    """
    sample_size = min(50, len(df))
    sample_df = df.sample(n=sample_size, random_state=RANDOM_SEED).copy()

    if "manual_label_normalized" not in sample_df.columns:
        sample_df["manual_label_normalized"] = None

    sample_df.to_csv(VALIDATION_SAMPLE_PATH, index=False, encoding="utf-8-sig")

    vader_acc = calculate_accuracy(
        sample_df["manual_label_normalized"], sample_df["vader_label"]
    )
    transformer_acc = calculate_accuracy(
        sample_df["manual_label_normalized"], sample_df["transformer_label"]
    )

    agreement = float((sample_df["vader_label"] == sample_df["transformer_label"]).mean())

    lines = [
        "# 情感模型验证报告（VADER vs Transformer）",
        "",
        "## 1. 验证设置",
        f"- 抽样数量：{sample_size}",
        f"- 随机种子：{RANDOM_SEED}",
        f"- 样本文件：`{VALIDATION_SAMPLE_PATH}`",
        "",
        "## 2. 准确率结果（基于人工标签）",
    ]

    if vader_acc is None or transformer_acc is None:
        lines.extend(
            [
                "- 当前样本缺少可用人工标签，无法计算准确率。",
                "- 已导出抽样文件，请在 `manual_label_normalized` 或新增 `manual_label` 列进行人工标注后复跑。",
            ]
        )
    else:
        lines.extend(
            [
                f"- VADER 准确率：**{vader_acc:.4f}**",
                f"- Transformer 准确率：**{transformer_acc:.4f}**",
            ]
        )

    lines.extend(
        [
            "",
            "## 3. 模型差异观察",
            f"- 两模型标签一致率：**{agreement:.4f}**",
            "- VADER 优点：速度快、可解释性强、部署轻量；对社交媒体语气词和标点敏感。",
            "- VADER 局限：难处理上下文依赖、复杂否定和隐喻语义。",
            "- Transformer 优点：上下文语义建模能力强，对复杂语句稳定性更好。",
            "- Transformer 局限：资源占用更高，推理耗时更长，且默认二分类模型对中性样本不够友好。",
            "",
            "## 4. 项目建议",
            "- 若追求在线高吞吐和低成本，可优先使用 VADER。",
            "- 若追求更高语义理解能力，建议优先使用 Transformer，并在业务数据上做进一步微调。",
        ]
    )

    VALIDATION_REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"[OK] 验证报告已保存: {VALIDATION_REPORT_PATH}")

    return {
        "vader_accuracy": vader_acc,
        "transformer_accuracy": transformer_acc,
        "model_agreement": agreement,
    }


def tokenize_for_keywords(text: str) -> List[str]:
    """用于负评关键词统计的轻量分词。"""
    if not isinstance(text, str):
        return []
    tokens = re.findall(r"[a-zA-Z]{2,}", text.lower())
    custom_stopwords = {
        "just",
        "really",
        "very",
        "get",
        "got",
        "still",
        "one",
        "two",
        "would",
        "could",
        "also",
        "im",
        "ive",
        "dont",
        "didnt",
        "cant",
        "wont",
    }
    stopwords = ENGLISH_STOP_WORDS.union(custom_stopwords)
    return [t for t in tokens if t not in stopwords and len(t) > 2]


def compute_topic_summary(
    df: pd.DataFrame,
    topic_col: str,
    text_col: str,
    final_label_col: str,
    final_score_col: str,
) -> pd.DataFrame:
    """按主题聚合情感统计并输出主题汇总表。"""
    rows = []
    grouped = df.groupby(topic_col, dropna=False)

    for topic, group in grouped:
        total = len(group)
        pos_cnt = int((group[final_label_col] == "positive").sum())
        neg_cnt = int((group[final_label_col] == "negative").sum())
        neu_cnt = int((group[final_label_col] == "neutral").sum())

        pos_pct = pos_cnt / total if total else 0.0
        neg_pct = neg_cnt / total if total else 0.0
        neu_pct = neu_cnt / total if total else 0.0

        mean_score = float(group[final_score_col].mean()) if total else 0.0
        nps_like = (pos_pct - neg_pct) * 100.0

        neg_texts = group.loc[group[final_label_col] == "negative", text_col].astype(str).tolist()
        token_counter = Counter()
        for text in neg_texts:
            token_counter.update(tokenize_for_keywords(text))

        top_keywords = "; ".join([w for w, _ in token_counter.most_common(10)])

        rows.append(
            {
                "topic": topic,
                "total_reviews": total,
                "positive_count": pos_cnt,
                "negative_count": neg_cnt,
                "neutral_count": neu_cnt,
                "positive_ratio": round(pos_pct, 4),
                "negative_ratio": round(neg_pct, 4),
                "neutral_ratio": round(neu_pct, 4),
                "avg_sentiment_score": round(mean_score, 4),
                "nps_like_score": round(nps_like, 2),
                "negative_top_keywords": top_keywords,
            }
        )

    summary_df = pd.DataFrame(rows).sort_values(by="nps_like_score", ascending=False)
    summary_df.to_csv(TOPIC_SUMMARY_PATH, index=False, encoding="utf-8-sig")
    print(f"[OK] 主题情感汇总已保存: {TOPIC_SUMMARY_PATH}")
    return summary_df


def create_visualizations(
    df: pd.DataFrame,
    topic_summary: pd.DataFrame,
    topic_col: str,
    text_col: str,
    final_label_col: str,
    metrics: Dict[str, Optional[float]],
) -> None:
    """生成学术报告级可视化图表。"""
    sns.set_theme(style="whitegrid")
    plt.rcParams["figure.figsize"] = (10, 6)
    plt.rcParams["axes.unicode_minus"] = False

    # 1) 整体情感分布饼图
    overall_counts = df[final_label_col].value_counts().reindex(
        ["positive", "neutral", "negative"], fill_value=0
    )
    plt.figure(figsize=(8, 8))
    plt.pie(
        overall_counts.values,
        labels=overall_counts.index,
        autopct="%1.1f%%",
        startangle=120,
    )
    plt.title("Overall Sentiment Distribution")
    plt.tight_layout()
    plt.savefig(VIZ_DIR / "1_overall_sentiment_distribution_pie.png", dpi=300)
    plt.close()

    # 2) 各主题情感分布分组柱状图
    plot_df = topic_summary.copy()
    top_topics = plot_df.nlargest(12, "total_reviews")["topic"].tolist()
    grouped_data = df[df[topic_col].isin(top_topics)].copy()
    grouped_data = (
        grouped_data.groupby([topic_col, final_label_col]).size().reset_index(name="count")
    )
    total_per_topic = grouped_data.groupby(topic_col)["count"].transform("sum")
    grouped_data["ratio"] = grouped_data["count"] / total_per_topic

    plt.figure(figsize=(14, 7))
    sns.barplot(data=grouped_data, x=topic_col, y="ratio", hue=final_label_col)
    plt.title("Sentiment Distribution by Topic")
    plt.xlabel("Topic")
    plt.ylabel("Ratio")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(VIZ_DIR / "2_topic_sentiment_grouped_bar.png", dpi=300)
    plt.close()

    # 3) 主题平均情感得分排序柱状图
    sorted_scores = topic_summary.sort_values(by="avg_sentiment_score", ascending=False)
    plt.figure(figsize=(14, 7))
    sns.barplot(data=sorted_scores, x="topic", y="avg_sentiment_score", color="#4C72B0")
    plt.title("Average Sentiment Score by Topic")
    plt.xlabel("Topic")
    plt.ylabel("Average Sentiment Score")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(VIZ_DIR / "3_topic_avg_sentiment_score_ranked_bar.png", dpi=300)
    plt.close()

    # 4) 痛点最高 3 个主题的负评词云
    worst_topics = topic_summary.nsmallest(3, "nps_like_score")["topic"].tolist()
    for idx, topic in enumerate(worst_topics, start=1):
        neg_text = " ".join(
            df.loc[
                (df[topic_col] == topic) & (df[final_label_col] == "negative"),
                text_col,
            ]
            .fillna("")
            .astype(str)
            .tolist()
        )
        if not neg_text.strip():
            neg_text = "no negative keywords"

        wc = WordCloud(
            width=1200,
            height=600,
            background_color="white",
            collocations=False,
        ).generate(neg_text)
        plt.figure(figsize=(12, 6))
        plt.imshow(wc, interpolation="bilinear")
        plt.axis("off")
        plt.title(f"Negative Word Cloud - Topic: {topic}")
        plt.tight_layout()
        plt.savefig(VIZ_DIR / f"4_wordcloud_painpoint_topic_{idx}.png", dpi=300)
        plt.close()

    # 5) VADER 与 Transformer 模型效果对比图
    compare_names = ["VADER", "Transformer", "Agreement"]
    compare_values = [
        metrics["vader_accuracy"] if metrics["vader_accuracy"] is not None else 0.0,
        metrics["transformer_accuracy"] if metrics["transformer_accuracy"] is not None else 0.0,
        metrics["model_agreement"] if metrics["model_agreement"] is not None else 0.0,
    ]

    plt.figure(figsize=(8, 6))
    compare_df = pd.DataFrame({"model": compare_names, "score": compare_values})
    sns.barplot(data=compare_df, x="model", y="score", hue="model", palette="Set2", legend=False)
    plt.ylim(0, 1)
    plt.title("Model Performance Comparison")
    plt.ylabel("Score (0-1)")
    for i, value in enumerate(compare_values):
        plt.text(i, value + 0.02, f"{value:.3f}", ha="center")
    plt.tight_layout()
    plt.savefig(VIZ_DIR / "5_vader_vs_transformer_comparison.png", dpi=300)
    plt.close()

    print(f"[OK] 可视化图表已输出至: {VIZ_DIR}")


def generate_business_report(
    topic_summary: pd.DataFrame,
    model_metrics: Dict[str, Optional[float]],
) -> None:
    """生成中文业务洞察与结论报告。"""
    top3 = topic_summary.nlargest(3, "nps_like_score")
    bottom3 = topic_summary.nsmallest(3, "nps_like_score")

    chosen_model = "Transformer"
    reason = "语义理解能力更强，适合复杂评论语境"
    va = model_metrics.get("vader_accuracy")
    ta = model_metrics.get("transformer_accuracy")
    if va is not None and ta is not None and va > ta:
        chosen_model = "VADER"
        reason = "在当前样本验证中准确率更高，且部署成本更低"

    lines = [
        "# 情感分析业务洞察报告",
        "",
        "## 1. 关键发现",
        "### 满意度最高的 3 个主题（可放大优势）",
    ]

    for _, row in top3.iterrows():
        lines.append(
            f"- **{row['topic']}**：NPS-like={row['nps_like_score']}，"
            f"平均情感分数={row['avg_sentiment_score']}。"
        )

    lines.extend(
        [
            "",
            "### 满意度最低的 3 个主题（客服优先优化痛点）",
        ]
    )
    for _, row in bottom3.iterrows():
        lines.append(
            f"- **{row['topic']}**：NPS-like={row['nps_like_score']}，"
            f"负评高频词={row['negative_top_keywords'] or '无明显高频词'}。"
        )

    lines.extend(
        [
            "",
            "## 2. 可落地改进建议",
            "### 营销侧建议",
            "- 对高满意主题加大投放预算与素材曝光，突出用户正向反馈关键词。",
            "- 在广告与详情页中复用高满意主题的核心卖点，强化差异化定位。",
            "- 结合高满意主题构建捆绑营销策略，提升交叉销售转化率。",
            "",
            "### 客服与运营侧建议",
            "- 对低满意主题建立专项工单队列，设置更短的首次响应 SLA。",
            "- 针对负评高频词设计标准化应答模板与补偿策略，降低投诉升级率。",
            "- 将低满意主题纳入周度复盘机制，跟踪负评占比与问题闭环时长。",
            "",
            "## 3. 模型方案选择",
            f"- 推荐最终方案：**{chosen_model}**。",
            f"- 选择理由：{reason}。",
            "- 实施建议：保留双模型并行监控机制，按月抽样评估漂移并更新阈值/策略。",
        ]
    )

    BUSINESS_REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"[OK] 业务洞察报告已保存: {BUSINESS_REPORT_PATH}")


def select_final_model_label(df: pd.DataFrame, metrics: Dict[str, Optional[float]]) -> pd.DataFrame:
    """
    根据验证结果选择最终用于业务产出的模型输出。
    - 默认优先 Transformer
    - 若 VADER 在可用准确率上更优，则切换 VADER
    """
    result_df = df.copy()

    vader_acc = metrics.get("vader_accuracy")
    transformer_acc = metrics.get("transformer_accuracy")
    use_vader = False

    if vader_acc is not None and transformer_acc is not None and vader_acc > transformer_acc:
        use_vader = True

    if use_vader:
        result_df["final_sentiment_label"] = result_df["vader_label"]
        result_df["final_sentiment_score"] = result_df["vader_compound"]
        chosen = "VADER"
    else:
        result_df["final_sentiment_label"] = result_df["transformer_label"]
        result_df["final_sentiment_score"] = result_df["transformer_score"]
        chosen = "Transformer"

    print(f"[INFO] 主题级业务分析采用模型: {chosen}")
    return result_df


def infer_training_label(df: pd.DataFrame) -> pd.Series:
    """
    推断可用于监督训练的标签列，并统一为字符串标签。
    优先级：
    1) manual_label_normalized
    2) sentiment_label
    3) label(数值/文本自动映射)
    """
    if "manual_label_normalized" in df.columns:
        label_series = df["manual_label_normalized"].copy()
        valid_count = int(label_series.notna().sum())
        if valid_count > 0:
            return label_series.astype(str).str.lower().replace("none", np.nan)

    if "sentiment_label" in df.columns:
        label_series = df["sentiment_label"].apply(normalize_sentiment_label)
        if int(label_series.notna().sum()) > 0:
            return label_series

    if "label" in df.columns:
        raw = df["label"]
        if pd.api.types.is_numeric_dtype(raw):
            uniques = set(pd.Series(raw.dropna().astype(int)).unique().tolist())
            mapping = {}
            if uniques.issubset({0, 1}):
                mapping = {0: "negative", 1: "positive"}
            elif uniques.issubset({-1, 1}):
                mapping = {-1: "negative", 1: "positive"}
            else:
                # 无法可靠映射时，直接转为字符串类别
                return raw.astype(str)
            return raw.map(mapping)
        return raw.astype(str).str.lower()

    raise KeyError("未找到可用于 BERT 训练的标签列（manual_label/sentiment_label/label）。")


def train_bert_model(
    df: pd.DataFrame,
    text_col: str,
    epochs: int = 2,
    train_batch_size: int = 16,
    eval_batch_size: int = 32,
    max_length: int = 128,
    bert_model_name: str = "bert-base-uncased",
    early_stopping_patience: int = 2,
    early_stopping_threshold: float = 0.0,
) -> Dict[str, float]:
    """
    使用 BERT 进行监督训练并记录训练数据。
    输出：
    - 训练好的模型与 tokenizer（models/）
    - 训练日志（reports/）
    - 评估指标（reports/）
    - 验证集预测（data/）
    """
    try:
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
            EarlyStoppingCallback,
            Trainer,
            TrainingArguments,
        )
    except Exception as exc:
        raise RuntimeError(
            "BERT 训练依赖缺失，请安装 transformers、datasets、accelerate、torch。"
        ) from exc

    work_df = df.copy()
    work_df[text_col] = work_df[text_col].fillna("").astype(str)
    work_df = work_df[work_df[text_col].str.strip() != ""].copy()
    work_df["bert_label_text"] = infer_training_label(work_df)
    work_df = work_df[work_df["bert_label_text"].notna()].copy()

    if work_df.empty:
        raise ValueError("BERT 训练数据为空，请检查标签列是否有效。")

    label_texts = sorted(work_df["bert_label_text"].astype(str).unique().tolist())
    if len(label_texts) < 2:
        raise ValueError(f"BERT 至少需要 2 个类别，当前仅检测到: {label_texts}")

    label2id = {label: idx for idx, label in enumerate(label_texts)}
    id2label = {idx: label for label, idx in label2id.items()}

    work_df["bert_label_id"] = work_df["bert_label_text"].map(label2id).astype(int)

    train_df, eval_df = train_test_split(
        work_df[[text_col, "bert_label_id", "bert_label_text"]],
        test_size=0.2,
        random_state=RANDOM_SEED,
        stratify=work_df["bert_label_id"],
    )

    tokenizer = AutoTokenizer.from_pretrained(bert_model_name)

    class ReviewDataset(torch.utils.data.Dataset):
        """轻量 Dataset 封装，适配 Trainer。"""

        def __init__(self, texts: List[str], labels: List[int]) -> None:
            self.texts = texts
            self.labels = labels

        def __len__(self) -> int:
            return len(self.texts)

        def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
            encoded = tokenizer(
                self.texts[idx],
                truncation=True,
                padding="max_length",
                max_length=max_length,
                return_tensors="pt",
            )
            item = {k: v.squeeze(0) for k, v in encoded.items()}
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
            return item

    train_dataset = ReviewDataset(
        texts=train_df[text_col].tolist(),
        labels=train_df["bert_label_id"].tolist(),
    )
    eval_dataset = ReviewDataset(
        texts=eval_df[text_col].tolist(),
        labels=eval_df["bert_label_id"].tolist(),
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        bert_model_name,
        num_labels=len(label_texts),
        id2label=id2label,
        label2id=label2id,
    )

    def compute_metrics(eval_pred) -> Dict[str, float]:
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=1)
        return {
            "accuracy": float(accuracy_score(labels, preds)),
            "precision_weighted": float(
                precision_score(labels, preds, average="weighted", zero_division=0)
            ),
            "recall_weighted": float(
                recall_score(labels, preds, average="weighted", zero_division=0)
            ),
            "f1_weighted": float(f1_score(labels, preds, average="weighted", zero_division=0)),
        }

    training_args_common = {
        "output_dir": str(MODEL_DIR / "bert_checkpoints"),
        "num_train_epochs": epochs,
        "per_device_train_batch_size": train_batch_size,
        "per_device_eval_batch_size": eval_batch_size,
        "learning_rate": 2e-5,
        "warmup_steps": 0,
        "weight_decay": 0.01,
        "save_strategy": "epoch",
        "logging_strategy": "steps",
        "logging_steps": 50,
        "load_best_model_at_end": True,
        "metric_for_best_model": "f1_weighted",
        "greater_is_better": True,
        "seed": RANDOM_SEED,
        "report_to": [],
    }

    # 兼容不同 transformers 版本：
    # - 新版: eval_strategy
    # - 旧版: evaluation_strategy
    try:
        training_args = TrainingArguments(
            **training_args_common,
            eval_strategy="epoch",
        )
    except TypeError:
        training_args = TrainingArguments(
            **training_args_common,
            evaluation_strategy="epoch",
        )

    trainer_common = {
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "compute_metrics": compute_metrics,
        "callbacks": [
            EarlyStoppingCallback(
                early_stopping_patience=early_stopping_patience,
                early_stopping_threshold=early_stopping_threshold,
            )
        ],
    }

    # 兼容不同 transformers 版本：
    # - 旧版: tokenizer=...
    # - 新版: processing_class=...
    try:
        trainer = Trainer(
            **trainer_common,
            tokenizer=tokenizer,
        )
    except TypeError:
        trainer = Trainer(
            **trainer_common,
            processing_class=tokenizer,
        )

    trainer.train()
    eval_metrics = trainer.evaluate()

    # 保存最终模型
    BERT_MODEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(BERT_MODEL_OUTPUT_DIR))
    tokenizer.save_pretrained(str(BERT_MODEL_OUTPUT_DIR))

    # 记录训练日志
    log_history = pd.DataFrame(trainer.state.log_history)
    if not log_history.empty:
        log_history.to_csv(BERT_TRAIN_LOG_PATH, index=False, encoding="utf-8-sig")
    else:
        pd.DataFrame([{"message": "no training log captured"}]).to_csv(
            BERT_TRAIN_LOG_PATH, index=False, encoding="utf-8-sig"
        )

    # 训练曲线图（loss / metric vs epoch）
    try:
        curve_df = log_history.copy()
        if not curve_df.empty and "epoch" in curve_df.columns:
            train_loss_df = curve_df[curve_df["loss"].notna()] if "loss" in curve_df.columns else pd.DataFrame()
            eval_loss_df = curve_df[curve_df["eval_loss"].notna()] if "eval_loss" in curve_df.columns else pd.DataFrame()

            fig, axes = plt.subplots(1, 2, figsize=(14, 5))

            # 左图：loss 曲线
            if not train_loss_df.empty:
                axes[0].plot(
                    train_loss_df["epoch"],
                    train_loss_df["loss"],
                    marker="o",
                    label="train_loss",
                )
            if not eval_loss_df.empty:
                axes[0].plot(
                    eval_loss_df["epoch"],
                    eval_loss_df["eval_loss"],
                    marker="s",
                    label="eval_loss",
                )
            axes[0].set_title("BERT Loss Curve")
            axes[0].set_xlabel("Epoch")
            axes[0].set_ylabel("Loss")
            axes[0].legend()
            axes[0].grid(alpha=0.3)

            # 右图：metric 曲线
            metric_cols = [
                "eval_accuracy",
                "eval_f1_weighted",
                "eval_precision_weighted",
                "eval_recall_weighted",
            ]
            metric_drawn = False
            for metric_col in metric_cols:
                if metric_col in curve_df.columns:
                    metric_df = curve_df[curve_df[metric_col].notna()]
                    if not metric_df.empty:
                        axes[1].plot(
                            metric_df["epoch"],
                            metric_df[metric_col],
                            marker="o",
                            label=metric_col,
                        )
                        metric_drawn = True
            axes[1].set_title("BERT Eval Metrics Curve")
            axes[1].set_xlabel("Epoch")
            axes[1].set_ylabel("Score")
            axes[1].set_ylim(0, 1.05)
            if metric_drawn:
                axes[1].legend()
            axes[1].grid(alpha=0.3)

            plt.tight_layout()
            plt.savefig(BERT_TRAIN_CURVE_PATH, dpi=300)
            plt.close()
            print(f"[OK] BERT 训练曲线图已保存: {BERT_TRAIN_CURVE_PATH}")
    except Exception as curve_exc:
        print(f"[WARN] 生成训练曲线图失败: {curve_exc}")

    # 保存评估指标与标签映射
    final_metrics = {
        "train_samples": int(len(train_df)),
        "eval_samples": int(len(eval_df)),
        "label_mapping": label2id,
        "model_name": bert_model_name,
        "num_epochs": int(epochs),
        "early_stopping_patience": int(early_stopping_patience),
        "early_stopping_threshold": float(early_stopping_threshold),
        "eval_loss": float(eval_metrics.get("eval_loss", np.nan)),
        "eval_accuracy": float(eval_metrics.get("eval_accuracy", np.nan)),
        "eval_precision_weighted": float(
            eval_metrics.get("eval_precision_weighted", np.nan)
        ),
        "eval_recall_weighted": float(eval_metrics.get("eval_recall_weighted", np.nan)),
        "eval_f1_weighted": float(eval_metrics.get("eval_f1_weighted", np.nan)),
    }
    BERT_METRICS_PATH.write_text(
        json.dumps(final_metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # 保存验证集预测明细
    pred_output = trainer.predict(eval_dataset)
    pred_ids = np.argmax(pred_output.predictions, axis=1)
    probs = torch.softmax(torch.tensor(pred_output.predictions), dim=1).numpy()
    pred_conf = probs.max(axis=1)

    pred_df = eval_df.copy().reset_index(drop=True)
    pred_df["bert_pred_id"] = pred_ids
    pred_df["bert_pred_label"] = pred_df["bert_pred_id"].map(id2label)
    pred_df["bert_pred_confidence"] = pred_conf
    pred_df.to_csv(BERT_EVAL_PREDICTIONS_PATH, index=False, encoding="utf-8-sig")

    print(f"[OK] BERT 模型已保存: {BERT_MODEL_OUTPUT_DIR}")
    print(f"[OK] BERT 训练日志已保存: {BERT_TRAIN_LOG_PATH}")
    print(f"[OK] BERT 评估指标已保存: {BERT_METRICS_PATH}")
    print(f"[OK] BERT 验证预测已保存: {BERT_EVAL_PREDICTIONS_PATH}")

    return final_metrics


def slugify_identifier(value: str) -> str:
    """将模型名等标识转为适合作为文件名的 slug。"""
    return re.sub(r"[^a-zA-Z0-9]+", "_", value).strip("_").lower()


def normalize_dataset1_label_row(row: pd.Series) -> Optional[str]:
    """
    针对 Dataset1.xlsx 的标签清洗逻辑。
    优先使用数值 label：
    - 0 -> negative
    - 1 -> neutral
    - 2 -> positive
    若 label 异常，则回退到 sentiment_label。
    """
    raw_label = row.get("label")
    raw_sentiment = row.get("sentiment_label")

    if pd.notna(raw_label):
        raw_text = str(raw_label).strip().lower()
        numeric_mapping = {
            "0": "negative",
            "1": "neutral",
            "2": "positive",
            "-1": "negative",
            "negative": "negative",
            "neutral": "neutral",
            "positive": "positive",
        }
        mapped = numeric_mapping.get(raw_text)
        if mapped is not None:
            return mapped

    return normalize_sentiment_label(raw_sentiment)


def load_baseline1_dataset(
    input_path: Path = BASELINE1_INPUT_PATH,
) -> Tuple[pd.DataFrame, str]:
    """加载并清洗 Dataset1.xlsx，返回标准化后的数据与文本列名。"""
    if not input_path.exists():
        raise FileNotFoundError(f"未找到 Baseline Model 1 数据文件: {input_path}")

    try:
        df = pd.read_excel(input_path)
    except Exception as exc:
        raise RuntimeError(f"读取 Excel 失败: {input_path}，错误: {exc}") from exc

    text_col = choose_column(
        df.columns.tolist(),
        ["text_clean", "clean_text", "processed_text", "text", "review", "comment"],
    )
    if text_col is None:
        raise KeyError("Dataset1.xlsx 中未找到文本字段。")

    df[text_col] = df[text_col].fillna("").astype(str)
    df = df[df[text_col].str.strip() != ""].copy()
    df["baseline1_label_text"] = df.apply(normalize_dataset1_label_row, axis=1)
    df = df[df["baseline1_label_text"].notna()].copy()

    if df.empty:
        raise ValueError("Dataset1.xlsx 清洗后无有效样本。")

    df.to_csv(BASELINE1_CLEAN_PATH, index=False, encoding="utf-8-sig")
    print(f"[OK] Baseline1 清洗数据已保存: {BASELINE1_CLEAN_PATH}")
    print(f"[INFO] Baseline1 标签分布: {df['baseline1_label_text'].value_counts().to_dict()}")
    return df, text_col


def load_baseline2_dataset(
    input_path: Path = BASELINE2_INPUT_PATH,
) -> Tuple[pd.DataFrame, str]:
    """加载并清洗 Dataset2.xlsx，返回标准化后的数据与文本列名。"""
    if not input_path.exists():
        raise FileNotFoundError(f"未找到 Baseline Model 2 数据文件: {input_path}")

    try:
        df = pd.read_excel(input_path)
    except Exception as exc:
        raise RuntimeError(f"读取 Excel 失败: {input_path}，错误: {exc}") from exc

    text_col = choose_column(
        df.columns.tolist(),
        ["text_clean", "clean_text", "processed_text", "text", "review", "comment"],
    )
    if text_col is None:
        raise KeyError("Dataset2.xlsx 中未找到文本字段。")

    df[text_col] = df[text_col].fillna("").astype(str)
    df = df[df[text_col].str.strip() != ""].copy()
    df["baseline2_label_text"] = df.apply(normalize_dataset1_label_row, axis=1)
    df = df[df["baseline2_label_text"].notna()].copy()

    if df.empty:
        raise ValueError("Dataset2.xlsx 清洗后无有效样本。")

    df.to_csv(BASELINE2_CLEAN_PATH, index=False, encoding="utf-8-sig")
    print(f"[OK] Baseline2 清洗数据已保存: {BASELINE2_CLEAN_PATH}")
    print(f"[INFO] Baseline2 标签分布: {df['baseline2_label_text'].value_counts().to_dict()}")
    return df, text_col


def split_baseline1_train_test(
    df: pd.DataFrame,
    text_col: str,
    test_size: float = 0.2,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, int], Dict[int, str]]:
    """按 8:2 分层切分训练集与测试集，并保存到 data 目录。"""
    label_texts = ["negative", "neutral", "positive"]
    label2id = {label: idx for idx, label in enumerate(label_texts)}
    id2label = {idx: label for label, idx in label2id.items()}

    work_df = df.copy()
    work_df["baseline1_label_id"] = work_df["baseline1_label_text"].map(label2id)
    work_df = work_df[work_df["baseline1_label_id"].notna()].copy()
    work_df["baseline1_label_id"] = work_df["baseline1_label_id"].astype(int)

    train_df, test_df = train_test_split(
        work_df,
        test_size=test_size,
        random_state=RANDOM_SEED,
        stratify=work_df["baseline1_label_id"],
    )

    train_df.to_csv(BASELINE1_TRAIN_PATH, index=False, encoding="utf-8-sig")
    test_df.to_csv(BASELINE1_TEST_PATH, index=False, encoding="utf-8-sig")
    print(f"[OK] Baseline1 训练集已保存: {BASELINE1_TRAIN_PATH}")
    print(f"[OK] Baseline1 测试集已保存: {BASELINE1_TEST_PATH}")

    plt.figure(figsize=(10, 6))
    plot_df = pd.concat(
        [
            train_df["baseline1_label_text"].value_counts().rename("count").reset_index().assign(split="train"),
            test_df["baseline1_label_text"].value_counts().rename("count").reset_index().assign(split="test"),
        ],
        ignore_index=True,
    )
    plot_df.columns = ["label", "count", "split"]
    sns.barplot(data=plot_df, x="label", y="count", hue="split")
    plt.title("Baseline Model 1 Train/Test Label Distribution")
    plt.xlabel("Label")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(BASELINE1_VIZ_SPLIT_PATH, dpi=300)
    plt.close()
    print(f"[OK] Baseline1 训练/测试集分布图已保存: {BASELINE1_VIZ_SPLIT_PATH}")

    return train_df, test_df, label2id, id2label


def split_baseline2_train_test(
    df: pd.DataFrame,
    text_col: str,
    test_size: float = 0.2,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, int], Dict[int, str]]:
    """按 8:2 分层切分 Baseline2 训练集与测试集，并保存到 data 目录。"""
    label_texts = ["negative", "neutral", "positive"]
    label2id = {label: idx for idx, label in enumerate(label_texts)}
    id2label = {idx: label for label, idx in label2id.items()}

    work_df = df.copy()
    work_df["baseline2_label_id"] = work_df["baseline2_label_text"].map(label2id)
    work_df = work_df[work_df["baseline2_label_id"].notna()].copy()
    work_df["baseline2_label_id"] = work_df["baseline2_label_id"].astype(int)

    train_df, test_df = train_test_split(
        work_df,
        test_size=test_size,
        random_state=RANDOM_SEED,
        stratify=work_df["baseline2_label_id"],
    )

    train_df.to_csv(BASELINE2_TRAIN_PATH, index=False, encoding="utf-8-sig")
    test_df.to_csv(BASELINE2_TEST_PATH, index=False, encoding="utf-8-sig")
    print(f"[OK] Baseline2 训练集已保存: {BASELINE2_TRAIN_PATH}")
    print(f"[OK] Baseline2 测试集已保存: {BASELINE2_TEST_PATH}")

    plt.figure(figsize=(10, 6))
    plot_df = pd.concat(
        [
            train_df["baseline2_label_text"].value_counts().rename("count").reset_index().assign(split="train"),
            test_df["baseline2_label_text"].value_counts().rename("count").reset_index().assign(split="test"),
        ],
        ignore_index=True,
    )
    plot_df.columns = ["label", "count", "split"]
    sns.barplot(data=plot_df, x="label", y="count", hue="split")
    plt.title("Baseline Model 2 Train/Test Label Distribution")
    plt.xlabel("Label")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(BASELINE2_VIZ_SPLIT_PATH, dpi=300)
    plt.close()
    print(f"[OK] Baseline2 训练/测试集分布图已保存: {BASELINE2_VIZ_SPLIT_PATH}")

    return train_df, test_df, label2id, id2label


def build_word_vocab(
    texts: List[str],
    max_vocab_size: int = 20000,
    min_freq: int = 1,
) -> Dict[str, int]:
    """基于训练文本构建词表，仅供 CNN/LSTM 使用。"""
    counter = Counter()
    for text in texts:
        counter.update(str(text).split())

    vocab = {
        TRADITIONAL_PAD_TOKEN: 0,
        TRADITIONAL_UNK_TOKEN: 1,
    }

    sorted_tokens = [
        token
        for token, freq in counter.most_common(max_vocab_size)
        if freq >= min_freq
    ]
    for token in sorted_tokens:
        if token not in vocab:
            vocab[token] = len(vocab)
    return vocab


def encode_text_to_ids(text: str, vocab: Dict[str, int], max_length: int) -> List[int]:
    """将文本编码成定长 token id 序列。"""
    tokens = str(text).split()
    token_ids = [vocab.get(token, vocab[TRADITIONAL_UNK_TOKEN]) for token in tokens[:max_length]]
    if len(token_ids) < max_length:
        token_ids.extend([vocab[TRADITIONAL_PAD_TOKEN]] * (max_length - len(token_ids)))
    return token_ids


class SequenceTextDataset(torch.utils.data.Dataset):
    """传统深度学习文本分类数据集。"""

    def __init__(
        self,
        texts: List[str],
        labels: List[int],
        vocab: Dict[str, int],
        max_length: int,
    ) -> None:
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        input_ids = encode_text_to_ids(self.texts[idx], self.vocab, self.max_length)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }


class TextCNNClassifier(nn.Module):
    """经典 TextCNN 文本分类模型。"""

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        num_classes: int,
        pad_idx: int,
        num_filters: int = 128,
        kernel_sizes: Tuple[int, ...] = (3, 4, 5),
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.convs = nn.ModuleList(
            [nn.Conv1d(embed_dim, num_filters, kernel_size=k) for k in kernel_sizes]
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters * len(kernel_sizes), num_classes)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(input_ids).transpose(1, 2)
        pooled_features = []
        for conv in self.convs:
            conv_out = torch.relu(conv(embedded))
            pooled = torch.max(conv_out, dim=2).values
            pooled_features.append(pooled)
        features = torch.cat(pooled_features, dim=1)
        return self.fc(self.dropout(features))


class LSTMTextClassifier(nn.Module):
    """双向 LSTM 文本分类模型。"""

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        hidden_dim: int,
        num_classes: int,
        pad_idx: int,
        num_layers: int = 1,
        dropout: float = 0.3,
        bidirectional: bool = True,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=lstm_dropout,
            bidirectional=bidirectional,
        )
        direction_factor = 2 if bidirectional else 1
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * direction_factor, num_classes)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(input_ids)
        _, (hidden_state, _) = self.lstm(embedded)
        if self.lstm.bidirectional:
            features = torch.cat((hidden_state[-2], hidden_state[-1]), dim=1)
        else:
            features = hidden_state[-1]
        return self.fc(self.dropout(features))


def evaluate_traditional_classifier(
    model: nn.Module,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device,
    criterion: nn.Module,
) -> Dict[str, object]:
    """评估传统深度学习模型，并返回损失、指标和预测细节。"""
    model.eval()
    losses = []
    all_labels: List[int] = []
    all_preds: List[int] = []
    all_probs: List[np.ndarray] = []

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            logits = model(input_ids)
            loss = criterion(logits, labels)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)

            losses.append(float(loss.item()))
            all_labels.extend(labels.cpu().numpy().tolist())
            all_preds.extend(preds.cpu().numpy().tolist())
            all_probs.extend(probs.cpu().numpy())

    avg_loss = float(np.mean(losses)) if losses else np.nan
    metrics = {
        "eval_loss": avg_loss,
        "accuracy": float(accuracy_score(all_labels, all_preds)),
        "precision_weighted": float(
            precision_score(all_labels, all_preds, average="weighted", zero_division=0)
        ),
        "recall_weighted": float(
            recall_score(all_labels, all_preds, average="weighted", zero_division=0)
        ),
        "f1_weighted": float(
            f1_score(all_labels, all_preds, average="weighted", zero_division=0)
        ),
        "pred_ids": all_preds,
        "pred_probs": all_probs,
    }
    return metrics


def train_traditional_deep_learning_experiment(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    text_col: str,
    label_id_col: str,
    label2id: Dict[str, int],
    id2label: Dict[int, str],
    model_key: str,
    experiment_prefix: str,
    epochs: int = 8,
    train_batch_size: int = 32,
    eval_batch_size: int = 64,
    max_length: int = 128,
    early_stopping_patience: int = 2,
    learning_rate: float = 1e-3,
) -> Dict[str, object]:
    """训练 CNN/LSTM 并返回测试集评估结果。"""
    safe_key = slugify_identifier(model_key)
    model_output_dir = MODEL_DIR / experiment_prefix / safe_key
    pred_path = DATA_DIR / f"{experiment_prefix}_{safe_key}_test_predictions.csv"
    log_path = REPORT_DIR / f"{experiment_prefix}_{safe_key}_training_log.csv"
    curve_path = VIZ_DIR / f"{experiment_prefix}_{safe_key}_training_curve.png"

    vocab = build_word_vocab(train_df[text_col].astype(str).tolist())
    train_dataset = SequenceTextDataset(
        texts=train_df[text_col].astype(str).tolist(),
        labels=train_df[label_id_col].astype(int).tolist(),
        vocab=vocab,
        max_length=max_length,
    )
    test_dataset = SequenceTextDataset(
        texts=test_df[text_col].astype(str).tolist(),
        labels=test_df[label_id_col].astype(int).tolist(),
        vocab=vocab,
        max_length=max_length,
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocab_size = len(vocab)
    num_classes = len(label2id)
    pad_idx = vocab[TRADITIONAL_PAD_TOKEN]

    if model_key == "cnn":
        model = TextCNNClassifier(
            vocab_size=vocab_size,
            embed_dim=128,
            num_classes=num_classes,
            pad_idx=pad_idx,
        )
        model_name = "TextCNN"
    elif model_key == "lstm":
        model = LSTMTextClassifier(
            vocab_size=vocab_size,
            embed_dim=128,
            hidden_dim=128,
            num_classes=num_classes,
            pad_idx=pad_idx,
        )
        model_name = "BiLSTM"
    else:
        raise KeyError(f"不支持的传统深度学习模型: {model_key}")

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    best_f1 = -1.0
    best_state_dict = None
    patience_counter = 0
    log_rows = []

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_losses = []

        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            logits = model(input_ids)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            epoch_losses.append(float(loss.item()))

        train_loss = float(np.mean(epoch_losses)) if epoch_losses else np.nan
        eval_metrics = evaluate_traditional_classifier(
            model=model,
            data_loader=test_loader,
            device=device,
            criterion=criterion,
        )
        current_f1 = float(eval_metrics["f1_weighted"])

        log_rows.append(
            {
                "epoch": epoch,
                "loss": train_loss,
                "eval_loss": float(eval_metrics["eval_loss"]),
                "eval_accuracy": float(eval_metrics["accuracy"]),
                "eval_precision_weighted": float(eval_metrics["precision_weighted"]),
                "eval_recall_weighted": float(eval_metrics["recall_weighted"]),
                "eval_f1_weighted": current_f1,
            }
        )

        if current_f1 > best_f1:
            best_f1 = current_f1
            best_state_dict = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                break

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
        model.to(device)

    final_eval = evaluate_traditional_classifier(
        model=model,
        data_loader=test_loader,
        device=device,
        criterion=criterion,
    )

    model_output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_key": model_key,
            "model_name": model_name,
            "model_state_dict": model.state_dict(),
            "vocab": vocab,
            "label2id": label2id,
            "id2label": id2label,
            "max_length": max_length,
        },
        model_output_dir / "model.pt",
    )

    log_df = pd.DataFrame(log_rows)
    log_df.to_csv(log_path, index=False, encoding="utf-8-sig")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(log_df["epoch"], log_df["loss"], marker="o", label="train_loss")
    axes[0].plot(log_df["epoch"], log_df["eval_loss"], marker="s", label="eval_loss")
    axes[0].set_title(f"{model_name} Loss Curve")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    metric_cols = [
        "eval_accuracy",
        "eval_f1_weighted",
        "eval_precision_weighted",
        "eval_recall_weighted",
    ]
    for metric_col in metric_cols:
        axes[1].plot(log_df["epoch"], log_df[metric_col], marker="o", label=metric_col)
    axes[1].set_title(f"{model_name} Eval Metrics Curve")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Score")
    axes[1].set_ylim(0, 1.05)
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(curve_path, dpi=300)
    plt.close()

    pred_ids = list(final_eval["pred_ids"])
    pred_probs = np.array(final_eval["pred_probs"])
    pred_conf = pred_probs.max(axis=1) if len(pred_probs) > 0 else np.array([])

    prediction_df = test_df.copy().reset_index(drop=True)
    prediction_df["pred_label_id"] = pred_ids
    prediction_df["pred_label"] = prediction_df["pred_label_id"].map(id2label)
    prediction_df["pred_confidence"] = pred_conf
    prediction_df.to_csv(pred_path, index=False, encoding="utf-8-sig")

    return {
        "model_key": model_key,
        "model_name": model_name,
        "train_samples": int(len(train_df)),
        "test_samples": int(len(test_df)),
        "epochs_requested": int(epochs),
        "eval_loss": float(final_eval["eval_loss"]),
        "accuracy": float(final_eval["accuracy"]),
        "precision_weighted": float(final_eval["precision_weighted"]),
        "recall_weighted": float(final_eval["recall_weighted"]),
        "f1_weighted": float(final_eval["f1_weighted"]),
        "prediction_path": str(pred_path),
        "curve_path": str(curve_path),
        "model_output_dir": str(model_output_dir),
    }


def train_transformer_experiment(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    text_col: str,
    label_id_col: str,
    label2id: Dict[str, int],
    id2label: Dict[int, str],
    model_key: str,
    model_name: str,
    experiment_prefix: str,
    epochs: int = 2,
    train_batch_size: int = 8,
    eval_batch_size: int = 16,
    max_length: int = 128,
    early_stopping_patience: int = 2,
    early_stopping_threshold: float = 0.0,
) -> Dict[str, object]:
    """训练单个 Transformer 模型并返回测试集评估结果。"""
    try:
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
            EarlyStoppingCallback,
            Trainer,
            TrainingArguments,
        )
    except Exception as exc:
        raise RuntimeError("Transformer 训练依赖缺失。") from exc

    safe_key = slugify_identifier(model_key)
    model_output_dir = MODEL_DIR / experiment_prefix / safe_key
    checkpoint_dir = MODEL_DIR / f"{experiment_prefix}_checkpoints" / safe_key
    pred_path = DATA_DIR / f"{experiment_prefix}_{safe_key}_test_predictions.csv"
    log_path = REPORT_DIR / f"{experiment_prefix}_{safe_key}_training_log.csv"
    curve_path = VIZ_DIR / f"{experiment_prefix}_{safe_key}_training_curve.png"

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    class ReviewDataset(torch.utils.data.Dataset):
        def __init__(self, texts: List[str], labels: List[int]) -> None:
            self.texts = texts
            self.labels = labels

        def __len__(self) -> int:
            return len(self.texts)

        def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
            encoded = tokenizer(
                self.texts[idx],
                truncation=True,
                padding="max_length",
                max_length=max_length,
                return_tensors="pt",
            )
            item = {k: v.squeeze(0) for k, v in encoded.items()}
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
            return item

    train_dataset = ReviewDataset(
        train_df[text_col].astype(str).tolist(),
        train_df[label_id_col].astype(int).tolist(),
    )
    test_dataset = ReviewDataset(
        test_df[text_col].astype(str).tolist(),
        test_df[label_id_col].astype(int).tolist(),
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id,
    )

    def compute_metrics(eval_pred) -> Dict[str, float]:
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=1)
        return {
            "accuracy": float(accuracy_score(labels, preds)),
            "precision_weighted": float(
                precision_score(labels, preds, average="weighted", zero_division=0)
            ),
            "recall_weighted": float(
                recall_score(labels, preds, average="weighted", zero_division=0)
            ),
            "f1_weighted": float(f1_score(labels, preds, average="weighted", zero_division=0)),
        }

    training_args_common = {
        "output_dir": str(checkpoint_dir),
        "num_train_epochs": epochs,
        "per_device_train_batch_size": train_batch_size,
        "per_device_eval_batch_size": eval_batch_size,
        "learning_rate": 2e-5,
        "warmup_steps": 0,
        "weight_decay": 0.01,
        "save_strategy": "epoch",
        "logging_strategy": "steps",
        "logging_steps": 50,
        "load_best_model_at_end": True,
        "metric_for_best_model": "f1_weighted",
        "greater_is_better": True,
        "seed": RANDOM_SEED,
        "report_to": [],
    }
    try:
        training_args = TrainingArguments(**training_args_common, eval_strategy="epoch")
    except TypeError:
        training_args = TrainingArguments(
            **training_args_common,
            evaluation_strategy="epoch",
        )

    trainer_common = {
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset,
        "eval_dataset": test_dataset,
        "compute_metrics": compute_metrics,
        "callbacks": [
            EarlyStoppingCallback(
                early_stopping_patience=early_stopping_patience,
                early_stopping_threshold=early_stopping_threshold,
            )
        ],
    }
    try:
        trainer = Trainer(**trainer_common, tokenizer=tokenizer)
    except TypeError:
        trainer = Trainer(**trainer_common, processing_class=tokenizer)

    trainer.train()
    eval_metrics = trainer.evaluate()

    model_output_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(model_output_dir))
    tokenizer.save_pretrained(str(model_output_dir))

    log_history = pd.DataFrame(trainer.state.log_history)
    if not log_history.empty:
        log_history.to_csv(log_path, index=False, encoding="utf-8-sig")

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        train_loss_df = log_history[log_history["loss"].notna()] if "loss" in log_history.columns else pd.DataFrame()
        eval_loss_df = log_history[log_history["eval_loss"].notna()] if "eval_loss" in log_history.columns else pd.DataFrame()
        if not train_loss_df.empty:
            axes[0].plot(train_loss_df["epoch"], train_loss_df["loss"], marker="o", label="train_loss")
        if not eval_loss_df.empty:
            axes[0].plot(eval_loss_df["epoch"], eval_loss_df["eval_loss"], marker="s", label="eval_loss")
        axes[0].set_title(f"{model_key} Loss Curve")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].legend()
        axes[0].grid(alpha=0.3)

        metric_cols = ["eval_accuracy", "eval_f1_weighted", "eval_precision_weighted", "eval_recall_weighted"]
        for metric_col in metric_cols:
            if metric_col in log_history.columns:
                metric_df = log_history[log_history[metric_col].notna()]
                if not metric_df.empty:
                    axes[1].plot(metric_df["epoch"], metric_df[metric_col], marker="o", label=metric_col)
        axes[1].set_title(f"{model_key} Eval Metrics Curve")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Score")
        axes[1].set_ylim(0, 1.05)
        axes[1].legend()
        axes[1].grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(curve_path, dpi=300)
        plt.close()

    pred_output = trainer.predict(test_dataset)
    pred_ids = np.argmax(pred_output.predictions, axis=1)
    pred_probs = torch.softmax(torch.tensor(pred_output.predictions), dim=1).numpy()
    pred_conf = pred_probs.max(axis=1)

    prediction_df = test_df.copy().reset_index(drop=True)
    prediction_df["pred_label_id"] = pred_ids
    prediction_df["pred_label"] = prediction_df["pred_label_id"].map(id2label)
    prediction_df["pred_confidence"] = pred_conf
    prediction_df.to_csv(pred_path, index=False, encoding="utf-8-sig")

    return {
        "model_key": model_key,
        "model_name": model_name,
        "train_samples": int(len(train_df)),
        "test_samples": int(len(test_df)),
        "epochs_requested": int(epochs),
        "eval_loss": float(eval_metrics.get("eval_loss", np.nan)),
        "accuracy": float(eval_metrics.get("eval_accuracy", np.nan)),
        "precision_weighted": float(eval_metrics.get("eval_precision_weighted", np.nan)),
        "recall_weighted": float(eval_metrics.get("eval_recall_weighted", np.nan)),
        "f1_weighted": float(eval_metrics.get("eval_f1_weighted", np.nan)),
        "prediction_path": str(pred_path),
        "curve_path": str(curve_path),
        "model_output_dir": str(model_output_dir),
    }


def build_baseline1_visualizations_and_report(
    results_df: pd.DataFrame,
    best_prediction_df: pd.DataFrame,
    id2label: Dict[int, str],
) -> None:
    """生成 Baseline Model 1 对比图与实验报告。"""
    metric_plot_df = results_df.melt(
        id_vars=["model_key"],
        value_vars=["accuracy", "precision_weighted", "recall_weighted", "f1_weighted"],
        var_name="metric",
        value_name="score",
    )

    plt.figure(figsize=(12, 6))
    sns.barplot(data=metric_plot_df, x="model_key", y="score", hue="metric")
    plt.title("Baseline Model 1 Deep Learning Model Comparison")
    plt.xlabel("Model")
    plt.ylabel("Score")
    plt.ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig(BASELINE1_VIZ_COMPARISON_PATH, dpi=300)
    plt.close()

    label_order = [id2label[idx] for idx in sorted(id2label.keys())]
    cm = confusion_matrix(
        best_prediction_df["baseline1_label_text"],
        best_prediction_df["pred_label"],
        labels=label_order,
    )
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=label_order, yticklabels=label_order)
    plt.title("Baseline Model 1 Best Model Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.savefig(BASELINE1_VIZ_CONFUSION_PATH, dpi=300)
    plt.close()

    best_row = results_df.sort_values(by="f1_weighted", ascending=False).iloc[0]
    report_lines = [
        "# Baseline Model 1 深度学习模型对比报告",
        "",
        "## 1. 实验设置",
        f"- 数据源：`{BASELINE1_INPUT_PATH}`",
        f"- 训练/测试划分：8:2（分层抽样，随机种子={RANDOM_SEED}）",
        f"- 训练集文件：`{BASELINE1_TRAIN_PATH}`",
        f"- 测试集文件：`{BASELINE1_TEST_PATH}`",
        "",
        "## 2. 模型对比结果",
    ]
    for _, row in results_df.sort_values(by="f1_weighted", ascending=False).iterrows():
        report_lines.append(
            f"- **{row['model_key']}**：Accuracy={row['accuracy']:.4f}，"
            f"Precision={row['precision_weighted']:.4f}，"
            f"Recall={row['recall_weighted']:.4f}，"
            f"F1={row['f1_weighted']:.4f}。"
        )
    report_lines.extend(
        [
            "",
            "## 3. 推荐结论",
            f"- 当前 Baseline Model 1 最优模型：**{best_row['model_key']}**。",
            f"- 推荐依据：其加权 F1 最高（{best_row['f1_weighted']:.4f}），适合三分类情感任务汇报展示。",
            f"- 对比图：`{BASELINE1_VIZ_COMPARISON_PATH}`",
            f"- 混淆矩阵：`{BASELINE1_VIZ_CONFUSION_PATH}`",
        ]
    )
    BASELINE1_REPORT_PATH.write_text("\n".join(report_lines), encoding="utf-8")
    print(f"[OK] Baseline1 实验报告已保存: {BASELINE1_REPORT_PATH}")


def build_baseline2_visualizations_and_report(
    results_df: pd.DataFrame,
    best_prediction_df: pd.DataFrame,
    id2label: Dict[int, str],
) -> None:
    """生成 Baseline Model 2 对比图与实验报告。"""
    metric_plot_df = results_df.melt(
        id_vars=["model_key"],
        value_vars=["accuracy", "precision_weighted", "recall_weighted", "f1_weighted"],
        var_name="metric",
        value_name="score",
    )

    plt.figure(figsize=(12, 6))
    sns.barplot(data=metric_plot_df, x="model_key", y="score", hue="metric")
    plt.title("Baseline Model 2 Deep Learning Model Comparison")
    plt.xlabel("Model")
    plt.ylabel("Score")
    plt.ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig(BASELINE2_VIZ_COMPARISON_PATH, dpi=300)
    plt.close()

    label_order = [id2label[idx] for idx in sorted(id2label.keys())]
    cm = confusion_matrix(
        best_prediction_df["baseline2_label_text"],
        best_prediction_df["pred_label"],
        labels=label_order,
    )
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=label_order, yticklabels=label_order)
    plt.title("Baseline Model 2 Best Model Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.savefig(BASELINE2_VIZ_CONFUSION_PATH, dpi=300)
    plt.close()

    best_row = results_df.sort_values(by="f1_weighted", ascending=False).iloc[0]
    report_lines = [
        "# Baseline Model 2 深度学习模型对比报告",
        "",
        "## 1. 实验设置",
        f"- 数据源：`{BASELINE2_INPUT_PATH}`",
        f"- 训练/测试划分：8:2（分层抽样，随机种子={RANDOM_SEED}）",
        f"- 训练集文件：`{BASELINE2_TRAIN_PATH}`",
        f"- 测试集文件：`{BASELINE2_TEST_PATH}`",
        "- 说明：当前 `Dataset2.xlsx` 中未发现单独命名的 AI 标签列，本实验按现有标准化情感标签进行全量监督学习评估。",
        "",
        "## 2. 模型对比结果",
    ]
    for _, row in results_df.sort_values(by="f1_weighted", ascending=False).iterrows():
        report_lines.append(
            f"- **{row['model_key']}**：Accuracy={row['accuracy']:.4f}，"
            f"Precision={row['precision_weighted']:.4f}，"
            f"Recall={row['recall_weighted']:.4f}，"
            f"F1={row['f1_weighted']:.4f}。"
        )
    report_lines.extend(
        [
            "",
            "## 3. 推荐结论",
            f"- 当前 Baseline Model 2 最优模型：**{best_row['model_key']}**。",
            f"- 推荐依据：其加权 F1 最高（{best_row['f1_weighted']:.4f}），可作为 Baseline Model 2 的主要汇报结果。",
            f"- 对比图：`{BASELINE2_VIZ_COMPARISON_PATH}`",
            f"- 混淆矩阵：`{BASELINE2_VIZ_CONFUSION_PATH}`",
        ]
    )
    BASELINE2_REPORT_PATH.write_text("\n".join(report_lines), encoding="utf-8")
    print(f"[OK] Baseline2 实验报告已保存: {BASELINE2_REPORT_PATH}")


def merge_experiment_results(
    existing_path: Path,
    new_results_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    合并实验结果，避免后续仅跑部分模型时覆盖历史结果。
    规则：
    - 若已有同名 model_key，则用最新结果覆盖
    - 若历史中存在其他模型结果，则保留
    """
    if existing_path.exists():
        try:
            existing_df = pd.read_csv(existing_path)
        except Exception:
            existing_df = pd.DataFrame()
    else:
        existing_df = pd.DataFrame()

    if existing_df.empty:
        merged_df = new_results_df.copy()
    else:
        merged_df = pd.concat([existing_df, new_results_df], ignore_index=True)
        if "model_key" in merged_df.columns:
            merged_df = merged_df.drop_duplicates(subset=["model_key"], keep="last")

    if "f1_weighted" in merged_df.columns:
        merged_df = merged_df.sort_values(by="f1_weighted", ascending=False).reset_index(drop=True)
    return merged_df


def run_baseline_model_1_experiment(
    input_path: Path = BASELINE1_INPUT_PATH,
    model_keys: Optional[List[str]] = None,
    epochs: int = 2,
    train_batch_size: int = 8,
    eval_batch_size: int = 16,
    max_length: int = 128,
    early_stopping_patience: int = 2,
    early_stopping_threshold: float = 0.0,
    preview_sample_size: Optional[int] = None,
) -> pd.DataFrame:
    """运行 Baseline Model 1：手工标注数据 8:2 划分 + 多模型深度学习对比。"""
    ensure_directories()
    set_random_seed(RANDOM_SEED)

    if model_keys is None or len(model_keys) == 0:
        model_keys = ["distilbert", "bert", "albert"]

    invalid_keys = [key for key in model_keys if key not in BASELINE1_MODEL_REGISTRY]
    if invalid_keys:
        raise KeyError(f"存在未注册的模型键: {invalid_keys}，可选: {list(BASELINE1_MODEL_REGISTRY)}")

    df, text_col = load_baseline1_dataset(input_path=input_path)
    if preview_sample_size is not None and preview_sample_size > 0:
        sample_size = min(preview_sample_size, len(df))
        original_total = len(df)
        sampled_groups = []
        for _, group_df in df.groupby("baseline1_label_text"):
            group_target = max(1, int(round(sample_size * len(group_df) / original_total)))
            sampled_groups.append(
                group_df.sample(min(group_target, len(group_df)), random_state=RANDOM_SEED)
            )
        df = pd.concat(sampled_groups, ignore_index=True)
        if len(df) > sample_size:
            df = df.sample(sample_size, random_state=RANDOM_SEED).reset_index(drop=True)
        print(f"[INFO] Baseline1 已启用预览模式，样本数: {len(df)}")

    train_df, test_df, label2id, id2label = split_baseline1_train_test(df=df, text_col=text_col)

    results = []
    for model_key in model_keys:
        model_name = BASELINE1_MODEL_REGISTRY[model_key]
        print(f"[INFO] Baseline1 正在训练模型: {model_key} -> {model_name}")
        if model_key in TRADITIONAL_DEEP_LEARNING_MODELS:
            result = train_traditional_deep_learning_experiment(
                train_df=train_df,
                test_df=test_df,
                text_col=text_col,
                label_id_col="baseline1_label_id",
                label2id=label2id,
                id2label=id2label,
                model_key=model_key,
                experiment_prefix="baseline1",
                epochs=max(epochs, 5),
                train_batch_size=max(train_batch_size, 32),
                eval_batch_size=max(eval_batch_size, 64),
                max_length=max_length,
                early_stopping_patience=early_stopping_patience,
            )
        else:
            result = train_transformer_experiment(
                train_df=train_df,
                test_df=test_df,
                text_col=text_col,
                label_id_col="baseline1_label_id",
                label2id=label2id,
                id2label=id2label,
                model_key=model_key,
                model_name=model_name,
                experiment_prefix="baseline1",
                epochs=epochs,
                train_batch_size=train_batch_size,
                eval_batch_size=eval_batch_size,
                max_length=max_length,
                early_stopping_patience=early_stopping_patience,
                early_stopping_threshold=early_stopping_threshold,
            )
        results.append(result)

    results_df = pd.DataFrame(results)
    results_df = merge_experiment_results(BASELINE1_RESULTS_PATH, results_df)
    results_df.to_csv(BASELINE1_RESULTS_PATH, index=False, encoding="utf-8-sig")
    print(f"[OK] Baseline1 模型对比结果已保存: {BASELINE1_RESULTS_PATH}")

    best_pred_path = Path(results_df.iloc[0]["prediction_path"])
    best_prediction_df = pd.read_csv(best_pred_path)
    build_baseline1_visualizations_and_report(
        results_df=results_df,
        best_prediction_df=best_prediction_df,
        id2label=id2label,
    )
    return results_df


def run_baseline_model_2_experiment(
    input_path: Path = BASELINE2_INPUT_PATH,
    model_keys: Optional[List[str]] = None,
    epochs: int = 2,
    train_batch_size: int = 8,
    eval_batch_size: int = 16,
    max_length: int = 128,
    early_stopping_patience: int = 2,
    early_stopping_threshold: float = 0.0,
) -> pd.DataFrame:
    """运行 Baseline Model 2：Dataset2.xlsx 全量 8:2 划分 + 多模型深度学习对比。"""
    ensure_directories()
    set_random_seed(RANDOM_SEED)

    if model_keys is None or len(model_keys) == 0:
        model_keys = ["distilbert", "bert", "albert"]

    invalid_keys = [key for key in model_keys if key not in BASELINE1_MODEL_REGISTRY]
    if invalid_keys:
        raise KeyError(f"存在未注册的模型键: {invalid_keys}，可选: {list(BASELINE1_MODEL_REGISTRY)}")

    df, text_col = load_baseline2_dataset(input_path=input_path)
    train_df, test_df, label2id, id2label = split_baseline2_train_test(df=df, text_col=text_col)

    results = []
    for model_key in model_keys:
        model_name = BASELINE1_MODEL_REGISTRY[model_key]
        print(f"[INFO] Baseline2 正在训练模型: {model_key} -> {model_name}")
        if model_key in TRADITIONAL_DEEP_LEARNING_MODELS:
            result = train_traditional_deep_learning_experiment(
                train_df=train_df,
                test_df=test_df,
                text_col=text_col,
                label_id_col="baseline2_label_id",
                label2id=label2id,
                id2label=id2label,
                model_key=model_key,
                experiment_prefix="baseline2",
                epochs=max(epochs, 5),
                train_batch_size=max(train_batch_size, 32),
                eval_batch_size=max(eval_batch_size, 64),
                max_length=max_length,
                early_stopping_patience=early_stopping_patience,
            )
        else:
            result = train_transformer_experiment(
                train_df=train_df,
                test_df=test_df,
                text_col=text_col,
                label_id_col="baseline2_label_id",
                label2id=label2id,
                id2label=id2label,
                model_key=model_key,
                model_name=model_name,
                experiment_prefix="baseline2",
                epochs=epochs,
                train_batch_size=train_batch_size,
                eval_batch_size=eval_batch_size,
                max_length=max_length,
                early_stopping_patience=early_stopping_patience,
                early_stopping_threshold=early_stopping_threshold,
            )
        results.append(result)

    results_df = pd.DataFrame(results)
    results_df = merge_experiment_results(BASELINE2_RESULTS_PATH, results_df)
    results_df.to_csv(BASELINE2_RESULTS_PATH, index=False, encoding="utf-8-sig")
    print(f"[OK] Baseline2 模型对比结果已保存: {BASELINE2_RESULTS_PATH}")

    best_pred_path = Path(results_df.iloc[0]["prediction_path"])
    best_prediction_df = pd.read_csv(best_pred_path)
    build_baseline2_visualizations_and_report(
        results_df=results_df,
        best_prediction_df=best_prediction_df,
        id2label=id2label,
    )
    return results_df


def run_pipeline(
    input_path: Path = DEFAULT_INPUT_PATH,
    enable_bert_training: bool = True,
    bert_epochs: int = 2,
    bert_early_stopping_patience: int = 2,
    bert_early_stopping_threshold: float = 0.0,
) -> None:
    """执行完整情感分析流程。"""
    set_random_seed(RANDOM_SEED)
    ensure_directories()

    df, text_col, topic_col, _ = load_data(input_path=input_path)

    # Step 1: VADER
    vader_df = vader_sentiment(df, text_col=text_col)

    # Step 2: Transformer
    merged_df = transformer_sentiment(vader_df, text_col=text_col)

    # Step 3: 验证报告
    metrics = build_validation_report(merged_df)

    # Step 4: 选择业务最终标签
    final_df = select_final_model_label(merged_df, metrics)

    # Step 5: 主题聚合统计
    topic_summary = compute_topic_summary(
        final_df,
        topic_col=topic_col,
        text_col=text_col,
        final_label_col="final_sentiment_label",
        final_score_col="final_sentiment_score",
    )

    # Step 6: 可视化
    create_visualizations(
        final_df,
        topic_summary=topic_summary,
        topic_col=topic_col,
        text_col=text_col,
        final_label_col="final_sentiment_label",
        metrics=metrics,
    )

    # Step 7: 业务报告
    generate_business_report(topic_summary, model_metrics=metrics)

    # Step 8: BERT 监督训练（可选）
    if enable_bert_training:
        train_bert_model(
            df=df,
            text_col=text_col,
            epochs=bert_epochs,
            early_stopping_patience=bert_early_stopping_patience,
            early_stopping_threshold=bert_early_stopping_threshold,
        )
    else:
        print("[INFO] 已按参数跳过 BERT 训练。")

    print("\n========== 全流程执行完成 ==========")
    print(f"- 校验数据: {VALIDATED_PATH}")
    print(f"- VADER结果: {VADER_RESULT_PATH}")
    print(f"- Transformer结果: {TRANSFORMER_RESULT_PATH}")
    print(f"- 主题汇总: {TOPIC_SUMMARY_PATH}")
    print(f"- 验证报告: {VALIDATION_REPORT_PATH}")
    print(f"- 业务报告: {BUSINESS_REPORT_PATH}")
    print(f"- 可视化目录: {VIZ_DIR}")
    if enable_bert_training:
        print(f"- BERT模型目录: {BERT_MODEL_OUTPUT_DIR}")
        print(f"- BERT训练日志: {BERT_TRAIN_LOG_PATH}")
        print(f"- BERT评估指标: {BERT_METRICS_PATH}")
        print(f"- BERT验证预测: {BERT_EVAL_PREDICTIONS_PATH}")
        print(f"- BERT训练曲线图: {BERT_TRAIN_CURVE_PATH}")


def main() -> None:
    """主函数，支持一键运行并提供异常提示。"""
    parser = argparse.ArgumentParser(description="CDS529 Sentiment Analysis Pipeline")
    parser.add_argument(
        "--skip-bert-training",
        action="store_true",
        help="跳过 BERT 监督训练步骤",
    )
    parser.add_argument(
        "--bert-epochs",
        type=int,
        default=2,
        help="BERT 训练轮数，默认 2",
    )
    parser.add_argument(
        "--bert-only",
        action="store_true",
        help="仅执行 BERT 监督训练，不运行 VADER/Transformer/可视化流程",
    )
    parser.add_argument(
        "--bert-early-stopping-patience",
        type=int,
        default=2,
        help="BERT Early Stopping 容忍轮数，默认 2",
    )
    parser.add_argument(
        "--bert-early-stopping-threshold",
        type=float,
        default=0.0,
        help="BERT Early Stopping 最小提升阈值，默认 0.0",
    )
    parser.add_argument(
        "--run-baseline1",
        action="store_true",
        help="运行 Dataset1.xlsx 的 Baseline Model 1 深度学习对比实验",
    )
    parser.add_argument(
        "--baseline1-models",
        type=str,
        default="distilbert,bert,albert",
        help="Baseline1 使用的模型键，逗号分隔，如 distilbert,bert,albert",
    )
    parser.add_argument(
        "--baseline1-epochs",
        type=int,
        default=2,
        help="Baseline1 每个模型训练轮数，默认 2",
    )
    parser.add_argument(
        "--baseline1-preview-sample-size",
        type=int,
        default=0,
        help="Baseline1 预览模式样本数，0 表示使用全量数据",
    )
    parser.add_argument(
        "--baseline1-early-stopping-patience",
        type=int,
        default=2,
        help="Baseline1 Early Stopping 容忍轮数，默认 2",
    )
    parser.add_argument(
        "--baseline1-early-stopping-threshold",
        type=float,
        default=0.0,
        help="Baseline1 Early Stopping 最小提升阈值，默认 0.0",
    )
    parser.add_argument(
        "--run-baseline2",
        action="store_true",
        help="运行 Dataset2.xlsx 的 Baseline Model 2 深度学习对比实验",
    )
    parser.add_argument(
        "--baseline2-models",
        type=str,
        default="distilbert,bert,albert",
        help="Baseline2 使用的模型键，逗号分隔，如 distilbert,bert,albert",
    )
    parser.add_argument(
        "--baseline2-epochs",
        type=int,
        default=2,
        help="Baseline2 每个模型训练轮数，默认 2",
    )
    parser.add_argument(
        "--baseline2-early-stopping-patience",
        type=int,
        default=2,
        help="Baseline2 Early Stopping 容忍轮数，默认 2",
    )
    parser.add_argument(
        "--baseline2-early-stopping-threshold",
        type=float,
        default=0.0,
        help="Baseline2 Early Stopping 最小提升阈值，默认 0.0",
    )
    args = parser.parse_args()

    try:
        if args.run_baseline2:
            baseline2_models = [
                item.strip()
                for item in args.baseline2_models.split(",")
                if item.strip()
            ]
            run_baseline_model_2_experiment(
                input_path=BASELINE2_INPUT_PATH,
                model_keys=baseline2_models,
                epochs=max(1, args.baseline2_epochs),
                early_stopping_patience=max(1, args.baseline2_early_stopping_patience),
                early_stopping_threshold=max(0.0, args.baseline2_early_stopping_threshold),
            )
        elif args.run_baseline1:
            baseline1_models = [
                item.strip()
                for item in args.baseline1_models.split(",")
                if item.strip()
            ]
            run_baseline_model_1_experiment(
                input_path=BASELINE1_INPUT_PATH,
                model_keys=baseline1_models,
                epochs=max(1, args.baseline1_epochs),
                early_stopping_patience=max(1, args.baseline1_early_stopping_patience),
                early_stopping_threshold=max(0.0, args.baseline1_early_stopping_threshold),
                preview_sample_size=(
                    args.baseline1_preview_sample_size
                    if args.baseline1_preview_sample_size > 0
                    else None
                ),
            )
        elif args.bert_only:
            set_random_seed(RANDOM_SEED)
            ensure_directories()
            df, text_col, _, _ = load_data(input_path=DEFAULT_INPUT_PATH)
            train_bert_model(
                df=df,
                text_col=text_col,
                epochs=max(1, args.bert_epochs),
                early_stopping_patience=max(1, args.bert_early_stopping_patience),
                early_stopping_threshold=max(0.0, args.bert_early_stopping_threshold),
            )
        else:
            run_pipeline(
                enable_bert_training=not args.skip_bert_training,
                bert_epochs=max(1, args.bert_epochs),
                bert_early_stopping_patience=max(1, args.bert_early_stopping_patience),
                bert_early_stopping_threshold=max(0.0, args.bert_early_stopping_threshold),
            )
    except Exception as exc:
        print("\n[ERROR] 流水线执行失败，请根据以下信息排查：")
        print(f"- 错误类型: {type(exc).__name__}")
        print(f"- 错误详情: {exc}")
        print("- 详细堆栈:")
        print(traceback.format_exc())


if __name__ == "__main__":
    main()
