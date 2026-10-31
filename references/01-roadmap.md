# Agent for MLE 带读路线（Steps 0–6）

> 目标：从零基础到能读懂、再能产出这个方向的论文。
> 配合[三遍读法](00-how-to-read-papers.md)使用。**按顺序读，别跳。**

## 先一句话定义这个领域

**Agent for MLE = 用 LLM 当"自动驾驶"的机器学习工程师**：给它一个数据集和任务描述，
它自己写代码（数据清洗、特征、建模、调参）、在环境里跑实验、读分数、然后决定下一步
改什么，循环往复，直到产出能拿奖牌/达标的解。

## 两流派总图（所有论文都落在这张图里）

```
             Agent for MLE
            /             \
   流派 A：无训练搜索      流派 B：在线训练
   （强 LLM + 搜索）       （训小模型 agent）
   AIDE / MLE-Bench       ML-Agent
   MLGym / AIRA
```

- **流派 A** 不训模型，靠强 LLM 的编码能力 + 搜索策略（树/进化）在代码空间里找解
- **流派 B** 用强化学习训一个小模型 agent，让它自己学会做 ML 工程
- 读完你会自己判断：哪条路走得通，它们的共同盲区在哪

---

## Step 0 — 地基（1 天，不读论文，先建直觉）

- **0.1 LLM Agent 是什么**：LLM + 工具调用 + "思考→行动→观察"循环（ReAct）+ 记忆。
  本质就是一个会写代码、能看报错、会改的循环体。
- **0.2 ML 工程闭环长啥样**：人类打 Kaggle = EDA → 特征 → 建模 → 调参 → 提交 → 看榜 → 再改。
  把每一步映射成"agent 能自动化的动作"。
- **0.3 锚定一句话**：Agent for MLE = 把这个闭环用 LLM 循环跑起来，且**能根据分数自我修正**。

## Step 1 — MLE-Bench（OpenAI，[arXiv 2410.07095](https://arxiv.org/abs/2410.07095)）：先看"考场"

**为什么先读它**：所有方法的成绩都在这一个考场上比，不懂考场就看不懂分数。

- **Pass 1 看**：摘要 + 图 1（75 场 Kaggle 概览）+ 奖牌定义部分
- **关键概念**：奖牌按 Kaggle 排行榜百分位定（铜/银/金）；`pass@k`（给 k 次机会的通过率）
- **带着啥问题读**：
  1. 一个 agent 怎么算"赢了一场竞赛"？
  2. 为什么用真实 Kaggle 而不是人造任务？（真实、有榜、难度有梯度）
- **记住**：75 场真实 Kaggle；最强组合 o1-preview+AIDE 拿 **16.9%** 任意奖牌率

📖 精读笔记：[mle-bench.md](papers/mle-bench.md)

## Step 2 — AIDE（Weco AI，[arXiv 2502.13138](https://arxiv.org/abs/2502.13138)）：看"冠军打法"⭐ 第一优先级精读

- **Pass 1 看**：摘要 + 问题形式化（把 ML 工程定义成"在代码空间 S 上最大化目标 h(s)"）+ 方法节
- **核心机制**：三种算子 **Draft / Debug / Improve** + 摘要算子（Summarization）；
  **搜索策略与编码能力解耦**；方案树为何替代线性聊天
- **带着啥问题读**：
  1. 为什么树比一条长聊天更顶用？
  2. o1-preview 套不套 AIDE 差多少？
- **记住（MLE-Bench Lite，o1-preview）**：任意奖牌 **7.6% → 36.4%**（近 5 倍）；均值超过 **51.38%** 的人类选手
- **代码**：[github.com/WecoAI/aideml](https://github.com/WecoAI/aideml)——精读后翻 `agent.py` 看 step 循环

📖 精读笔记：[aide.md](papers/aide.md)

## Step 3 — ML-Agent（上海交大，[arXiv 2505.23723](https://arxiv.org/abs/2505.23723)）：看"另一条路线"（流派 B 标杆）

- **三大件**：
  1. exploration-enriched fine-tuning（让小模型生成**多样动作**，丰富 RL 探索）
  2. step-wise RL（**单步训练**，加速经验收集）
  3. agentic ML-specific reward module（把杂乱 ML 反馈统一成标量奖励）
- **带着啥问题读**：
  1. 不训练 vs 训练，你站哪边？为什么？
  2. **承重前提**：reward module 真能把噪声信号变成稳定奖励吗？
- **记住**：7B Qwen-2.5，只在 **9 个** ML 任务上训练，却超过 671B 的 DeepSeek-R1 agent、媲美 GPT-5

📖 精读笔记：[ml-agent.md](papers/ml-agent.md)

## Step 4 — 综述拉宽（[arXiv 2503.21460](https://arxiv.org/abs/2503.21460)，Luo et al.）：建立全局坐标系

一句话框架：**Build**（怎么造 agent）→ **Collaborate**（多 agent 怎么协作）→ **Evolve**
（agent 怎么自我进化/学习）。前两步读的 AIDE 落在 Build、ML-Agent 落在 Evolve 两端。
以后看任何新论文，先把它塞进这个三维坐标，立刻知道它"新在哪"。

## Step 5 — AIRA（Meta，[arXiv 2507.02554](https://arxiv.org/abs/2507.02554)）：批判读，AIDE 的"照妖镜"

**为什么必读**：AIRA 保留 AIDE 的算子、只换搜索策略（贪心 vs MCTS vs 进化搜索），发现
**搜索算法几乎没用，算子集合才是瓶颈**；甚至关掉 AIDE 的 Memory 算子，奖牌率几乎不变。

- **带着啥问题读**：
  1. 如果算子比搜索重要，改进 agent 该把钱花在哪？
  2. 这直接挑战"树搜索很关键"的叙事——被挑战之后，AIDE 还剩什么是对的？
- **记住**：MLE-Bench Lite 奖牌率 39.6% → **47.7%**；量化了 **9–13% 的泛化缺口**（agent 在过拟合验证集）

📖 精读笔记：[aira.md](papers/aira.md)

## Step 6 — MLGym（Meta，[arXiv 2502.14499](https://arxiv.org/abs/2502.14499)）：看"可训练"的基准环境

- **核心贡献**：首个 ML Gym——把 agent 的动作空间标准化，从而可以直接上强化学习训练；
  13 个开放任务 + Docker 隔离 + **自主性六级量表**（L0 复现 → L5 长期议程）
- **关键发现**：前沿模型目前卡在 L1（改进基线主要靠调超参），**生成不出新假设/算法**
- **带着啥问题读**：
  1. "只测 L1"意味着对 L2+ 的什么完全没测？
  2. 如果让你设计 L1→L2 之间的保障机制，你会怎么做？
- **方法论彩蛋**：Gemini-1.5-Pro 用约 **1/9 的成本**达到 o1-preview 99% 的性能——
  做实验务必把 **cost** 当一等指标，别只看准确率

📖 精读笔记：[mlgym.md](papers/mlgym.md)

---

## 读完之后

五篇连成一条线：**MLE-Bench（考场）→ AIDE（打法）→ ML-Agent（训练派）→ AIRA（批判）→ MLGym（可训练环境）**。
它们的共同盲区：agent 自主但黑箱、不可审计——这往往就是下一个研究问题的入口。
