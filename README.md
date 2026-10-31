<div align="center">

# 🧭 Zero to MLE Agent

**从零读懂 "AI 机器学习工程师"——Agent for MLE 带读路线 + 论文精读笔记**

A curated learning path & in-depth paper notes for **Agent for MLE**
(LLM agents that automate machine learning engineering) — from zero background to research-ready.

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Papers](https://img.shields.io/badge/papers%20covered-5-blue)](#-论文精读笔记)
[![Status](https://img.shields.io/badge/status-actively%20updated-brightgreen)]()
[![语言](https://img.shields.io/badge/语言-中文%20%7C%20English-orange)](README.en.md)

</div>

---

## 这是什么

**Agent for MLE** = 用 LLM 当"自动驾驶的机器学习工程师"：给它一个数据集和任务描述，它自己写代码、跑实验、看分数、决定下一步改什么，循环往复直到达标（AIDE、MLE-Bench、MLGym、AIRA 都在这个方向上）。

这个仓库提供两样东西：

1. **一条带读路线**：零基础该按什么顺序读论文、每篇读到什么程度、带着什么问题读
2. **五篇精读笔记**：每篇按统一模板（问题 → 方法 → 结果 → **批判性阅读**）拆解，不只复述，还指出论文的"承重前提"和可攻击的软肋——这是写论文的弹药

> 不需要你发明新模型。这个方向本质是**把 ML 工程流程外包给一个会写代码、能看指标的 LLM 循环体**，你缺的只是阅读方法和几篇奠基论文的脉络——本仓库补齐这两样。

## 学习路线（6 步，别跳）

| 步骤 | 内容 | 读什么 | 详细 |
|---|---|---|---|
| **Step 0** | 地基：LLM Agent 是什么 + ML 工程闭环 | 不读论文，先建直觉 | — |
| **Step 1** | 先看"考场" | MLE-Bench (OpenAI) | [笔记](docs/papers/mle-bench.md) |
| **Step 2** | 看"冠军打法" ⭐ 第一优先级 | AIDE (Weco AI) | [笔记](docs/papers/aide.md) |
| **Step 3** | 看"另一条路线"（训练派） | ML-Agent (SJTU) | [笔记](docs/papers/ml-agent.md) |
| **Step 4** | 综述拉宽，建坐标系 | LLM Agent Survey | — |
| **Step 5** | 批判读：AIDE 的"照妖镜" | AIRA (Meta) | [笔记](docs/papers/aira.md) |
| **Step 6** | 看"可训练"的基准环境 | MLGym (Meta) | [笔记](docs/papers/mlgym.md) |

配套方法论：[论文三遍读法](docs/00-how-to-read-papers.md)（15 分钟 skim → 精读 → 批判读）。

## 📚 论文精读笔记

| 论文 | arXiv | 一句话收获 | 笔记 |
|---|---|---|---|
| **MLE-Bench** (OpenAI) | [2410.07095](https://arxiv.org/abs/2410.07095) | 75 场真实 Kaggle 当考场，奖牌率=头条指标；o1+AIDE 16.9% | [📖](docs/papers/mle-bench.md) |
| **AIDE** (Weco AI) | [2502.13138](https://arxiv.org/abs/2502.13138) | 把 ML 工程形式化为"代码空间上的树搜索"，Draft/Debug/Improve 三算子 | [📖](docs/papers/aide.md) |
| **ML-Agent** (SJTU) | [2505.23723](https://arxiv.org/abs/2505.23723) | 7B 模型在线 RL 训练，9 个任务上训练却媲美 GPT-5 | [📖](docs/papers/ml-agent.md) |
| **AIRA** (Meta) | [2507.02554](https://arxiv.org/abs/2507.02554) | 搜索算法几乎没用、**算子才是瓶颈**；量化了 9–13% 泛化缺口 | [📖](docs/papers/aira.md) |
| **MLGym** (Meta) | [2502.14499](https://arxiv.org/abs/2502.14499) | 首个"可训练"ML Gym + 自主性六级量表（L0 复现 → L5 长期议程） | [📖](docs/papers/mlgym.md) |

## 两流派一张图

```
             Agent for MLE
            /             \
   流派 A：无训练搜索      流派 B：在线训练
   （强 LLM + 搜索）       （训小模型 agent）
   AIDE / MLE-Bench       ML-Agent
   MLGym / AIRA
            \             /
        【共同盲区：自主但黑箱，不可审计】
        metric hacking / 过拟合验证集 / 无法审计为什么改代码
```

## 如何使用本仓库

1. 先读 [论文三遍读法](docs/00-how-to-read-papers.md)（10 分钟）
2. 按 [学习路线](docs/01-roadmap.md) 的顺序读论文，每篇配合精读笔记
3. 读每篇笔记时**先自己回答"带着啥问题读"**，再看答案
4. 欢迎提 PR 补充新论文笔记（模板见 [CONTRIBUTING](CONTRIBUTING.md)）

## Roadmap

- [x] 5 篇核心论文精读笔记
- [x] 零基础带读路线
- [ ] 综述笔记（LLM Agent Survey, arXiv 2503.21460）
- [ ] 多智能体通信 / 协作拓扑方向笔记
- [ ] 每月跟进新论文（欢迎 PR）

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=Zephyr-Song/zero-to-mle-agent&type=Date)](https://star-history.com/#Zephyr-Song/zero-to-mle-agent&Date)

## License

[MIT](LICENSE) © 2026 宋锦程 (Zephyr-Song)
