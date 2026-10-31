# AIRA — 搜索算法几乎没用，算子才是瓶颈

**[arXiv 2507.02554](https://arxiv.org/abs/2507.02554) · Meta FAIR + UCL · NeurIPS 2025 · 代码 [github.com/facebookresearch/aira-dojo](https://github.com/facebookresearch/aira-dojo)**
**类型**：方法/分析型（AIDE 的"照妖镜"）。本方向最重要的批判性论文。

## 1. 核心问题 / 动机
- 现有研究 agent（尤其 AIDE）把**搜索逻辑、代码算子、算力**纠缠在一起，难以定位"性能到底从哪里来"，也无法做大规模对照实验。
- 作者把 AI research agent 形式化为三元组：**(Search Policy, Operator Set, Fitness Function)**，从而能把"搜索算法"和"算子"解耦开来单独换、单独测。

## 2. 方法（解耦框架 + AIRA-dojo）
- **解耦**：保留 AIDE 的算子（Draft/Debug/Improve/Memory），只换 Search Policy（Greedy → MCTS → Evolutionary），看谁贡献大。
- **AIRA 算子改进**（相对 AIDE）：
  - **dynamic complexity cue**：按节点已有子节点数动态调节"先简单后复杂"，防过早过度工程。
  - **scoped Memory**：Draft/Improve 只看**兄弟节点**的记忆，而非全局日志 → 减 mode collapse。
  - **Crossover 算子**：把两个不同解的代码片段重组。
  - 对推理模型显式要求结构化思考，且思考过程在给其他算子看之前被剥离。
- **AIRA-dojo**：可扩展执行环境（容器隔离、固定 GPU/CPU/内存/时限配额、严格可复现）。

## 3. 关键结果（最该记住的三条）
- **搜索算法几乎没用，算子才是瓶颈**：MCTS/Evolutionary 相对 greedy 增益极小；**关掉 AIDE 的 Memory 算子，奖牌率几乎不变**。
- **SOTA 数字**：MLE-Bench Lite 奖牌率 **39.6%（AIDE）→ 47.7%（AIRA 最佳配对）**；最新 dojo 框架到 **55%**。
- **泛化缺口**：若 agent 按**测试集**而非验证集选解，奖牌率还能再涨 **9–13%**——说明现在 agent 在**过拟合验证指标**。

## 4. 批判性阅读
- **对后续研究的启示**：与其卷搜索算法，不如卷**算子设计、想法多样性、防过拟合**——这是被对照实验证明过的方向。
- **泛化缺口 = 可信性问题的量化证据**：agent 过拟合验证集是"自主但不可信"的温和版，说明需要外部审计/监督机制。
- **AIRA-dojo 的工程范式值得学**：固定配额 + 容器隔离 + 隐藏测试标签，是任何 ML agent 实验环境的三件套。
- **衍生必读**：同作者系 "What Does It Take to Be a Good AI Research Agent? Studying the Role of Ideation Diversity"（[arXiv 2511.15593](https://arxiv.org/abs/2511.15593)）——论证**想法多样性**与 MLE-bench 成绩正相关。

## 5. 和已读论文的关系
AIRA 是 AIDE 的受控消融：AIDE（树搜索叙事）→ AIRA（算子才是瓶颈）。
它把 MLE-Bench 当考场、把 dojo 当环境——读完你会重新评估 Step 2 里 AIDE 的"树"到底贡献了几成。
