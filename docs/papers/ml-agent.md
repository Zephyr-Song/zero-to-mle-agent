# ML-Agent — 用在线强化学习训练 ML 工程 Agent

**[arXiv 2505.23723](https://arxiv.org/abs/2505.23723) · 上海交大 · 代码 [github.com/MASWorks/ML-Agent](https://github.com/MASWorks/ML-Agent)**
**类型**：方法型（流派 B：在线训练）。和 AIDE（流派 A：无训练搜索）对照读。

## 1. 一句话
不靠强 LLM + 树搜索，而是**在线强化学习训练一个小模型 agent**，让它自己学会做 ML 工程。

## 2. 三大件（方法核心）
- **① exploration-enriched fine-tuning**：让小模型生成**多样动作**，丰富 RL 探索（避免过早收敛到单一策略）。
- **② step-wise RL（单步训练）**：不等问题跑完才给奖励，**每步**就训练 → 大幅加速经验收集。
- **③ agentic ML-specific reward module**：把杂乱 ML 反馈（准确率/损失/运行时间/报错）**统一成标量奖励**，喂给 RL。

## 3. 关键结果
- 7B Qwen-2.5，只在 **9 个** ML 任务上训练。
- 却超过 671B 的 DeepSeek-R1 agent、媲美 GPT-5，且**跨任务泛化**。
- 意义：agent 能力可以"训"出来，不一定全靠模型 scale——和 MLGym 的"可训练"方向一致。

## 4. 批判性阅读
- **承重前提**：reward module 真能把噪声 ML 信号稳定化成干净标量奖励吗？论文这块写得偏高层、没给公式/消融——审稿人最爱挑的软肋。
- 训练数据只有 9 个任务，"跨任务泛化"的边界需谨慎——泛化到 unseen 任务类型的证据不足。
- 同样只优化标量奖励，训练出的 agent 可能学会"刷奖励"而非"真做好 ML"——reward hacking 在训练派更严重。

## 5. 和已读论文的关系
- AIDE（流派 A，无训练搜索）vs ML-Agent（流派 B，在线训练）——两条路线的对照是本领域最基本的世界观。
- 两派共享同一个缺口：**都缺人类在环的可信保障**。
