<div align="center">

# 🧭 Zero to MLE Agent

**从零写一个 MLE Agent · Build Your Own MLE Agent**

一个 build-your-own 系列课程：每课一个可独立运行的 Python 文件，
从 60 行的 ReAct 循环开始，一步步造出一个能自己做机器学习的 Agent。

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Lessons](https://img.shields.io/badge/lessons-1%2F7-blue)](#-课程表)
[![Deps](https://img.shields.io/badge/起步依赖-零-success)]()
[![语言](https://img.shields.io/badge/语言-中文%20%7C%20English-orange)](README.en.md)

</div>

---

## 这是什么

**Agent for MLE** = 用 LLM 当"自动驾驶的机器学习工程师"：给它数据集和任务，它自己写代码、跑实验、看分数、决定下一步改什么（AIDE、MLE-Bench、MLGym 都是这个方向）。

与其先读十篇论文，不如**亲手造一个**。本系列把一个 MLE Agent 拆成 7 课，每课：

- ✅ **一个文件**：`lessons/l0X_xxx.py`，独立可跑，`python` 一下就看到 agent 在动
- ✅ **零依赖起步**：前几课纯标准库；**MockLLM 内置，不需要任何 API key**
- ✅ **可复现**：固定随机种子，每次运行输出一致
- ✅ **真踩坑写进注释**：课程代码里留下的 Bug 与修复过程，都是真实调试记录
- 🎯 **终点**：累计约 300 行，一个带方案树、搜索策略和人类审计闸门的最小 MLE Agent

## 📚 课程表

| 课 | 内容 | 状态 | 运行 |
|---|---|---|---|
| **L01** | ReAct 循环——60 行写出一个会推理的循环（Think→Act→Observe） | ✅ | `python lessons/l01_react_loop.py` |
| **L02** | 真实环境——agent 自己写 ML 代码，环境真实评分，拿到第一个准确率 | 🚧 | — |
| **L03** | 方案树与三算子——Draft / Debug / Improve | 📝 | — |
| **L04** | Summarization——上下文不爆炸的关键（AIDE 碾压 ReAct 的原因） | 📝 | — |
| **L05** | 搜索策略——greedy vs 树搜索，亲手验证 AIRA 的"算子才是瓶颈" | 📝 | — |
| **L06** | HITL 闸门——给 agent 加人类审计层（本系列的原创创新点） | 📝 | — |
| **L07** | 接真 LLM——把 MockLLM 换成 OpenAI 兼容接口，跑真实小基准 | 📝 | — |

> git 提交历史即课程进度：每一课一个 commit，`git log` 就是从零到一的全过程。

## ⚡ 快速开始

```bash
git clone https://github.com/Zephyr-Song/zero-to-mle-agent.git
cd zero-to-mle-agent
python lessons/l01_react_loop.py   # 看 agent 用 7 轮二分猜中数字并主动 FINISH
```

无需 GPU，无需 API key，无需安装任何东西（Python 3.10+ 即可）。

## 📖 参考资料（论文精读笔记）

造轮子读不懂原理时，来这里补理论——每篇按"问题 → 方法 → 结果 → 批判性阅读"拆解：

| 论文 | arXiv | 一句话 | 笔记 |
|---|---|---|---|
| MLE-Bench (OpenAI) | [2410.07095](https://arxiv.org/abs/2410.07095) | 75 场真实 Kaggle 当考场 | [📖](references/papers/mle-bench.md) |
| AIDE (Weco AI) | [2502.13138](https://arxiv.org/abs/2502.13138) | ML 工程 = 代码空间树搜索 | [📖](references/papers/aide.md) |
| ML-Agent (SJTU) | [2505.23723](https://arxiv.org/abs/2505.23723) | 7B 在线 RL 媲美 GPT-5 | [📖](references/papers/ml-agent.md) |
| AIRA (Meta) | [2507.02554](https://arxiv.org/abs/2507.02554) | 算子才是瓶颈，不是搜索 | [📖](references/papers/aira.md) |
| MLGym (Meta) | [2502.14499](https://arxiv.org/abs/2502.14499) | 自主性六级量表 L0–L5 | [📖](references/papers/mlgym.md) |

还有：[论文三遍读法](references/00-how-to-read-papers.md) · [完整带读路线](references/01-roadmap.md)

## Contributing

- 发现 Bug / 讲解不清楚 → 开 issue 或直接 PR
- 想认领某一课 → 见 [CONTRIBUTING](CONTRIBUTING.md)

## License

[MIT](LICENSE) © 2026 Zephyr-Song
