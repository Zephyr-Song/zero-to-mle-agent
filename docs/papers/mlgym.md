# MLGym — 首个"可训练"的 ML 工程 Gym

**[arXiv 2502.14499](https://arxiv.org/abs/2502.14499) · Meta GenAI + FAIR · 代码 [github.com/facebookresearch/MLGym](https://github.com/facebookresearch/MLGym)**
**类型**：基准/系统型论文（先框架后基准）。它也是最被记住的概念贡献——自主性六级量表——的出处。

## 1. 核心贡献：和 MLE-Bench / MLAgentBench 差在哪
对比维度：Gym Interface / Algorithmic Tasks / Open-Ended Research / Flexible Artifacts / Agentic Harness。
- MLE-Bench：75 场 Kaggle，**只能评测、不能训练**，任务多已被 SOTA 解决。
- MLAgentBench：能评测 + 有 harness，但**无 Gym 接口**，难接 RL。
- **MLGym 五项全√**：尤其 Gym Interface 意味着把动作空间标准化 → 可直接上强化学习训练（"可训练"的字面含义）。

## 2. 核心机制：四件套 + 解耦 + 隔离
系统拆成 **Agents / Environment / Datasets / Tasks** 四个可独立替换的模块，关键是 **Agent 与环境解耦**：agent 只发 bash 动作、收观察，环境在 Docker 里隔离执行（非 root 用户、权限受限）。

## 3. 概念核心：自主性六级量表（最该吃透的部分）
- **L0 复现**：复现已有论文结果。
- **L1 改进基线**：给定非 SOTA 基线代码，提升其性能。
- **L2 达 SOTA**：只给任务描述 + SOTA 出现前的公开文献，复现出 SOTA（等同人类解开放问题）。
- **L3 新科学贡献**：新方法在多榜达 SOTA，有顶会价值。
- **L4 突破贡献**：定义关键问题/方向，Oral / Best Paper 级。
- **L5 长期议程**：数周~数年持续产出，诺奖/图灵级。
- **关键发现**：MLGym-Bench 目前**只测 L1**，前沿模型主要靠**调超参**改进基线，**生成不出新假设/算法/架构**——"自主性天花板"与"可靠性瓶颈"是同一件事的两面。

## 4. 实验结果 + 方法论彩蛋
- 主结果：前沿模型能改进基线但主要靠更好超参；o1-preview 综合最强。
- 彩蛋：Gemini-1.5-Pro 用约 **1/9 成本**达 o1-preview **99%** 性能 → 做实验务必把 **cost/compute** 当一等指标，别只看准确率。

## 5. 批判性阅读
- **承重前提："Bench 只测 L1"** → 对 L2+ 的**科学新颖性完全没测**，论文自己也没解决"如何客观评价 novelty"——这是明确的开放问题。
- **L1→L2 之间缺一层保障**：agent 从"调参"跨到"提新假设"时，恰恰是最需要外部审计的时刻——这是后续研究（包括 human-in-the-loop 方向）的自然入口。
- 一句话定位：**MLGym 定义了 agent 能做多自主；"自主到哪一步该被兜住"仍是开放问题。**

## 6. 和已读论文的关系
MLE-Bench（考场）→ AIDE（打法）→ ML-Agent（训练派）→ AIRA（批判）→ **MLGym（可训练环境 + 自主性坐标系）**。
六级量表是给整个领域定的"尺子"，以后任何新论文都可以问一句：它做到了 L 几？
