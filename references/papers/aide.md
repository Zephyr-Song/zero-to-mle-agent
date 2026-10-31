# AIDE — 把 ML 工程变成"代码空间上的树搜索"

**[arXiv 2502.13138](https://arxiv.org/abs/2502.13138) · Weco AI · 代码 [github.com/WecoAI/aideml](https://github.com/WecoAI/aideml)**
**类型**：方法型论文（先形式化问题 → 方法 → 证伪）。读它顺便学"方法论文怎么写"。

## 1. 摘要的结构（1 段 4 句，每句一个功能——可套用）
- S1 背景+痛点：ML 进步大，但背后是复杂繁琐的 trial-and-error。
- S2 痛点具体化+亮方案：工程师大量时间耗在试错 → `we introduce AIDE`。
- S3 核心一句话：**frames ML engineering as a code optimization problem, trial-and-error as a tree search.**（全文最值钱的一句）
- S4 结果+范围+开源：Kaggle / MLE-Bench / RE-Bench SOTA + 代码链接。

## 2. Introduction 的 5 段骨架（教科书级，可整段套用）
- P1 动机+前辈天花板：AutoML/NAS/HPO 的死穴——"需预定义搜索空间、靠领域专家、偏暴力、易过拟合"。
- P2 LLM 机遇（转折）：从 config space 跳到 **code space** 直接搜索。
- P3 贡献+核心差异：AIDE 把历史解组织在**树结构**而非 ReACT 的线性历史。
- P4 第三方验证：搬 OpenAI MLE-Bench、METR RE-Bench 背书，比自吹硬。
- P5 论文路线：上半形式化、下半实证。

## 3. 形式化（全文地基）
- 批 ReACT 的 POMDP：缺分解问题的原则方法、历史全堆进 prompt 会爆。
- 核心公式：**s\* = argmax_{s∈S} h(s)**，S=代码解空间，h=无状态目标（准确率/损失）。
- 这一步把"ML 工程"翻译成"无状态优化"，树搜索才顺理成章。

## 4. 方法（四个设计件）
- **Search Policy π**：硬编码规则决定何时 Draft/Debug/Improve（先采多样初始解、持续改最优、限 debug 次数）。
- **Coding Operator f**：Drafting（从零写）/ Debugging（修 bug）/ Improving（**只改一个原子改动**让效果可测）。
- **Summarization Σ（最关键）**：不堆日志，只抽 metrics / hyperparameters / debug hints → 维持无状态视角、不爆 prompt。这是碾压 ReACT 的底层原因。

## 5. 评测（三基准证伪，不只报喜）
- Weco-Kaggle（63 赛）：AIDE 均值**超 51.38% 人类**，碾压 H2O AutoML(35.34%)、AutoGPT(32.34%)。
- MLE-Bench（75 赛）：o1+AIDE 任意奖牌 **7.6%→36.4%**、有效提交 63.6%→92.4%。
- RE-Bench（AI R&D）：6h 内超人类专家；**但主动坦白弱点**——大体量代码库环境不行、贪心策略易陷局部最优。

## 6. 写作上可直接偷师的三点
1. 引言 5 段骨架整段套：动机→机遇→贡献→验证预览→路线。
2. 先形式化再方法：先把问题"翻译"对，方法才站得住。
3. 评测主动写失败：诚实增信。

## 7. 批判性阅读
- **承重前提："树搜索 > 线性历史"**——后续 AIRA 证明**算子集合比搜索算法更关键**（关掉 Memory 算子奖牌率几乎不变），"树"叙事被大幅削弱。
- Summarization 把历史压成 metrics，意味着**无法审计 agent 为什么改某段代码**——黑箱问题的根源之一。
- 全程优化单一标量 h(s)，无视 metric hacking / 数据泄漏风险。
