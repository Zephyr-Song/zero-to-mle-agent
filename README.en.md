<div align="center">

# 🧭 Zero to MLE Agent

**Build Your Own MLE Agent — from a 60-line ReAct loop to a full mini agent.**

A build-your-own series: one runnable Python file per lesson, zero dependencies
to start, no API key needed (built-in mock LLM).

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Lessons](https://img.shields.io/badge/lessons-2%2F7-blue)](#-lessons)
[![Deps](https://img.shields.io/badge/starting%20deps-zero-success)]()
[![中文](https://img.shields.io/badge/%E4%B8%AD%E6%96%87%20README-here-orange)](README.md)

</div>

---

## What is this

**Agent for MLE** = an LLM acting as an *autonomous machine learning engineer*: it writes code, runs experiments, reads the metrics, and decides what to change next (AIDE, MLE-Bench, MLGym all live here).

Instead of reading ten papers first, **build one yourself**. 7 lessons, each:

- ✅ **One file** (`lessons/l0X_xxx.py`), runnable standalone: `python lessons/l01_react_loop.py`
- ✅ **Zero dependencies to start**; built-in **MockLLM — no API key needed**
- ✅ **Reproducible** (fixed seeds)
- ✅ **Real debugging stories in the comments**
- 🎯 **End state**: ~300 lines total — a minimal MLE agent with a solution tree, search policy, and a human-oversight gate

## 📚 Lessons

| # | Topic | Status | Run |
|---|---|---|---|
| **L01** | ReAct loop — a reasoning loop in ~60 lines | ✅ | `python lessons/l01_react_loop.py` |
| **L02** | A real environment — the agent writes ML code, runs sandboxed, gets a **real accuracy** | ✅ | `pip install scikit-learn` → `python lessons/l02_real_env.py` |
| **L03** | Solution tree & operators — Draft / Debug / Improve | 📝 | — |
| **L04** | Summarization — why AIDE beats linear ReAct | 📝 | — |
| **L05** | Search policy — greedy vs tree search (verify AIRA yourself) | 📝 | — |
| **L06** | HITL gate — add a human-oversight layer | 📝 | — |
| **L07** | Swap in a real LLM (OpenAI-compatible) | 📝 | — |

Git history = course progress: one commit per lesson.

## ⚡ Quick start

```bash
git clone https://github.com/Zephyr-Song/zero-to-mle-agent.git
cd zero-to-mle-agent
python lessons/l01_react_loop.py
```

No GPU, no API key, nothing to install (Python 3.10+).

## 📖 Reference: paper notes

Theory companion, in-depth notes with a critical-reading section each:
[MLE-Bench](references/papers/mle-bench.md) · [AIDE](references/papers/aide.md) · [ML-Agent](references/papers/ml-agent.md) · [AIRA](references/papers/aira.md) · [MLGym](references/papers/mlgym.md)
Also: [three-pass reading method](references/00-how-to-read-papers.md) · [full reading roadmap](references/01-roadmap.md)

## Contributing

Issues and PRs welcome — see [CONTRIBUTING](CONTRIBUTING.md).

## License

[MIT](LICENSE) © 2026 Zephyr-Song
