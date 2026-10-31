<div align="center">

# 🧭 Zero to MLE Agent

**A curated learning path & in-depth paper notes for Agent for MLE — from zero to research-ready.**

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Papers](https://img.shields.io/badge/papers%20covered-5-blue)](#-paper-notes)
[![Status](https://img.shields.io/badge/status-actively%20updated-brightgreen)]()
[![中文](https://img.shields.io/badge/%E4%B8%AD%E6%96%87%20README-here-orange)](README.md)

</div>

---

## What is this

**Agent for MLE** = using an LLM as an *autonomous machine learning engineer*: give it a dataset and a task, and it writes code, runs experiments, reads the metrics, and decides what to change next — looping until it hits the target (AIDE, MLE-Bench, MLGym, AIRA all live here).

This repo gives you two things:

1. **A guided reading path**: what order to read papers in, how deep to go on each, and what questions to hold while reading
2. **Five in-depth notes**: each paper dissected with a uniform template (Problem → Method → Results → **Critical reading**) — not summaries, but notes that expose each paper's load-bearing assumptions and weak spots. That's ammunition for writing your own paper.

## Reading path (6 steps)

| Step | Content | Paper | Notes |
|---|---|---|---|
| 0 | Foundations: what an LLM agent is + the ML engineering loop | — | — |
| 1 | The "exam hall" first | MLE-Bench (OpenAI) | [notes](docs/papers/mle-bench.md) |
| 2 | The champion's playbook ⭐ read first | AIDE (Weco AI) | [notes](docs/papers/aide.md) |
| 3 | The other route (training-based) | ML-Agent (SJTU) | [notes](docs/papers/ml-agent.md) |
| 4 | Broaden with a survey | LLM Agent Survey | — |
| 5 | Critical read: AIDE's mirror | AIRA (Meta) | [notes](docs/papers/aira.md) |
| 6 | A *trainable* benchmark environment | MLGym (Meta) | [notes](docs/papers/mlgym.md) |

Method: the [three-pass reading technique](docs/00-how-to-read-papers.md).

## 📚 Paper notes

| Paper | arXiv | One-line takeaway | Notes |
|---|---|---|---|
| **MLE-Bench** (OpenAI) | [2410.07095](https://arxiv.org/abs/2410.07095) | 75 real Kaggle comps as the exam; medal rate as headline metric | [📖](docs/papers/mle-bench.md) |
| **AIDE** (Weco AI) | [2502.13138](https://arxiv.org/abs/2502.13138) | ML engineering as tree search over the space of code | [📖](docs/papers/aide.md) |
| **ML-Agent** (SJTU) | [2505.23723](https://arxiv.org/abs/2505.23723) | 7B model + online RL, trained on 9 tasks, rivals GPT-5 | [📖](docs/papers/ml-agent.md) |
| **AIRA** (Meta) | [2507.02554](https://arxiv.org/abs/2507.02554) | Search policy barely matters — **operators are the bottleneck**; 9–13% generalization gap | [📖](docs/papers/aira.md) |
| **MLGym** (Meta) | [2502.14499](https://arxiv.org/abs/2502.14499) | First *trainable* ML Gym + a 6-level autonomy ladder (L0–L5) | [📖](docs/papers/mlgym.md) |

## Two camps, one picture

```
                Agent for MLE
               /             \
   Camp A: training-free    Camp B: online RL
   (strong LLM + search)    (train a small agent)
   AIDE / MLE-Bench         ML-Agent
   MLGym / AIRA
               \             /
      【Shared blind spot: autonomous but opaque, unauditable】
      metric hacking / validation overfitting / no "why did it change this"
```

## Contributing

PRs welcome — see [CONTRIBUTING](CONTRIBUTING.md) for the note template.

## License

[MIT](LICENSE) © 2026 Song Jincheng (Zephyr-Song)
