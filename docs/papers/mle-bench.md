# MLE-Bench — 评测 ML Agent 的"考场"

**[arXiv 2410.07095](https://arxiv.org/abs/2410.07095) · OpenAI · 代码 [github.com/openai/mle-bench](https://github.com/openai/mle-bench)**
**类型**：基准/评测论文（"考场"）。所有方法的成绩都在这一个考场上比——不懂它就看不懂任何论文的分数。

## 1. 核心问题 / 动机
- 现有基准**没有"整体性"衡量 AI agent 在自主端到端 ML 工程上的能力**（语言模型在代码基准上强，但"从零跑完一场 Kaggle"是另一回事）。
- 设计两原则：(i) 选有挑战、代表当代 ML 工程的任务；(ii) 能与**人类水平**比较。
- 意义：可作 OpenAI Preparedness / Anthropic RSP / DeepMind Frontier Safety 等"自主能力度量"。

## 2. 基准怎么造出来的
- **从 5673 个已结束 Kaggle 竞赛** → 排除社区赛剩 586 → 按 9 条标准（相关性/可复现 grading/可重建 split 等）人工筛选 → 最终 **75 场**。
- 复杂度：22 低 / 38 中 / 15 高；额外 7 场作 dev split；覆盖 15 个领域；原总奖金 $1.95M。
- **每个样本 = 一场 Kaggle**：竞赛描述 + 数据集（多数用**新建 train-test 划分**防记忆）+ 本地 grading code + **Private leaderboard 快照**（用于和人类排名比）。
- **环境**：Docker 容器，每场最多 **24h**，方法无关（只要求交一个 **CSV**），本地 validation server 只查合法性不给分；防作弊：日志分析 + 抄袭检测（相似度 >60% 判违规）。

## 3. 奖牌怎么定 + 关键指标
- 奖牌按 **Private leaderboard**（非 Public，防过拟合）排名，门槛逻辑与 Kaggle 一致（按队伍数 Top 40%/20%/10% 对应铜/银/金）。
- **头条指标 = Any Medal（拿到铜牌及以上）的尝试百分比**。天花板极高——史上仅 9 人拿过 75+ 场不同竞赛奖牌。
- **pass@k**：估计每场给 k 次尝试时的获牌率，`pass@k = E[1 − (n−c choose k)/(n choose k)]`。

## 4. 关键实证结果
- **最佳配置 o1-preview + AIDE**：Any Medal (pass@1) = **16.9%**。
  细分：Made Submission 98.4% / Valid 82.8% / Above Median 29.4% / Bronze 3.4% / Silver 4.1% / Gold 9.4%。
- **模型对比（都用 AIDE）**：o1-preview **16.9%** > GPT-4o 8.7% > Claude 3.5 Sonnet 7.6% > Llama-3.1-405B 3.0%。
- **脚手架对比（都用 GPT-4o）**：AIDE **8.7%** > OpenHands 4.4% > MLAB 0.8%。
- **扩展**：pass@8 翻倍到 **34.1%**；时间 24h→100h 提到 11.8%。

## 5. 作者自陈局限
1. **污染/抄袭风险**：模型预训练可能见过题解甚至测试集；仅证 GPT-4o 无系统污染。
2. **AI R&D 覆盖不全**：真实科研常无清晰问题定义，找数据集/指标本身是问题一部分。
3. **资源密集**：主实验 ≈ **1800 GPU hours**；单 seed 平均 127.5M 输入 tokens，难复现。

## 6. 批判性阅读（承重前提与软肋）
- **"Any Medal" 把难度差巨大的竞赛压成一个数字**：低复杂度 34.3% vs 高复杂度 10.0%，头标指标掩盖了分布——引用它时要分复杂度看。
- **方法无关（只交 CSV）= 只能评测不能训练**：这正是后来 MLGym 的出发点。
- **污染威胁基准长期有效性**：随训练数据覆盖更多 Kaggle，private leaderboard 比较会退化。

## 7. 和已读论文的关系
MLE-Bench（考场）→ AIDE（打法）→ MLGym（可训练）→ ML-Agent（训练派）→ AIRA（批判）。
