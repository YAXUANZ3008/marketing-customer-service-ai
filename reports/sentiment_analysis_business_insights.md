# 情感分析业务洞察报告

## 1. 关键发现
### 满意度最高的 3 个主题（可放大优势）
- **Sports & Outdoors**：NPS-like=45.21，平均情感分数=0.1912。
- **Automotive**：NPS-like=44.72，平均情感分数=0.1868。
- **Electronics**：NPS-like=44.63，平均情感分数=0.1853。

### 满意度最低的 3 个主题（客服优先优化痛点）
- **Home & Kitchen**：NPS-like=42.93，负评高频词=leave; met; lowest; ignoring; absolute; unique; way; expectations; regarding; question。
- **Clothing**：NPS-like=43.23，负评高频词=leave; met; lowest; ignoring; absolute; expectations; unique; way; question; ignored。
- **Electronics**：NPS-like=44.63，负评高频词=leave; unique; way; ignoring; met; absolute; lowest; expectations; basic; object。

## 2. 可落地改进建议
### 营销侧建议
- 对高满意主题加大投放预算与素材曝光，突出用户正向反馈关键词。
- 在广告与详情页中复用高满意主题的核心卖点，强化差异化定位。
- 结合高满意主题构建捆绑营销策略，提升交叉销售转化率。

### 客服与运营侧建议
- 对低满意主题建立专项工单队列，设置更短的首次响应 SLA。
- 针对负评高频词设计标准化应答模板与补偿策略，降低投诉升级率。
- 将低满意主题纳入周度复盘机制，跟踪负评占比与问题闭环时长。

## 3. 模型方案选择
- 推荐最终方案：**VADER**。
- 选择理由：在当前样本验证中准确率更高，且部署成本更低。
- 实施建议：保留双模型并行监控机制，按月抽样评估漂移并更新阈值/策略。