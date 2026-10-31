"""
L02 · 真实环境：agent 第一次写 ML 代码，拿到真实准确率
======================================================

跑法：pip install scikit-learn  →  python lessons/l02_real_env.py

【和 L01 的本质区别】
L01 的动作空间是"猜一个数"，观察是三个字；本课动作空间升级为
**"写一段完整的 Python 代码"**，观察是**真实评测分数**——
agent 写的代码会被环境真正执行、真正训练模型、在隐藏测试集上真正打分。

【连接你读过的论文】
- 交 CSV 契约 = MLE-Bench 的做法：agent 只需产出 submission.csv，环境负责评分。
  写不出文件 / 报错 / 超时 = 无效提交 = 0 分（MLE-Bench 里 17.2% 的提交就是这样废掉的）。
- 子进程隔离 = MLGym / AIRA-dojo 的"穷人版"：agent 代码跑在独立进程里，
  它崩溃、死循环、写出垃圾文件，都污染不了主程序。真产品用 Docker 做同样的事。
- 迭代换模型 = AIDE 的 Improve 算子的雏形：看分数、决定下一步改什么。

【关键设计决定】
环境把数据写成 train.csv（带 target）和 test.csv（不带），**测试标签永远不给 agent**——
agent 只能从"提交→得分"这个标量反馈里学习。这就是"黑箱优化"的起点，
也是后面 L04（上下文管理）和 L06（HITL 闸门）要解决的问题的根源。

【思考题】
1. 为什么 agent 代码要跑在子进程里，而不是 import 进主程序直接调用？
   （提示：如果 agent 写了个 while True: pass，主程序会怎样？）
2. score=0 的回合和"模型不好"的回合，观察里应该怎么区分？为什么这个区分很重要？
3. 环境 目前只返回一行"准确率"。如果把报错堆栈、训练日志全塞给 agent，
   是帮它还是害它？（这是 L04 Summarization 的伏笔）

【下一课预告】
L03：目前是"一条直线往上爬"（greedy）。下一课把所有历史解组织成**方案树**，
引入 AIDE 的三个算子——agent 可以回头、可以分叉、可以修 bug。
"""

import csv
import subprocess
import sys
import tempfile
from pathlib import Path

try:
    import numpy as np
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split
except ImportError:
    raise SystemExit("本课需要 scikit-learn：先运行  pip install scikit-learn")


# ---------- 1. 环境：一个真实评分的 ML 代码沙盒 ----------

class MLCodeEnv:
    """契约：agent 提交一段代码 → 代码读取 train.csv/test.csv → 训练 →
    写出 submission.csv（每行一个预测）→ 环境对隐藏测试集打分。"""

    def __init__(self, dataset: str = "breast_cancer", test_size: float = 0.3, seed: int = 42):
        data = load_breast_cancer()
        X_tr, X_te, y_tr, y_te = train_test_split(
            data.data, data.target, test_size=test_size,
            random_state=seed, stratify=data.target)
        self._y_test = y_te.astype(int)
        self.workspace = Path(tempfile.mkdtemp(prefix="l02_env_"))
        self._dump(self.workspace / "train.csv",
                   [f"f{i}" for i in range(X_tr.shape[1])] + ["target"],
                   [row + [int(t)] for row, t in zip(X_tr.tolist(), y_tr)])
        self._dump(self.workspace / "test.csv",
                   [f"f{i}" for i in range(X_te.shape[1])],
                   X_te.tolist())
        self.task = (
            f"数据集: {dataset}（{X_tr.shape[1]} 个特征，二分类）\n"
            f"工作区: {self.workspace}（含 train.csv / test.csv，可打开看看）\n"
            f"你的动作: 提交一段 Python 代码，读取上述两个文件，训练模型，"
            f"写出 submission.csv（每行一个 0/1 预测）。测试集标签你看不到。"
        )

    @staticmethod
    def _dump(path: Path, header: list, rows: list) -> None:
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(header)
            w.writerows(rows)

    def step(self, code: str) -> tuple:
        """执行 agent 的代码并真实评分。返回 (观察, 准确率)。"""
        (self.workspace / "solution.py").write_text(code, encoding="utf-8")
        try:
            subprocess.run([sys.executable, "solution.py"], cwd=self.workspace,
                           timeout=120, capture_output=True, text=True)
        except subprocess.TimeoutExpired:
            return "子进程超时（>120s）——无效提交", 0.0
        sub = self.workspace / "submission.csv"
        if not sub.exists():
            return "没有产出 submission.csv——无效提交", 0.0
        pred = np.loadtxt(sub, delimiter=",")
        if pred.shape[0] != len(self._y_test):
            return f"预测数量 {pred.shape[0]} != 测试集 {len(self._y_test)}——无效提交", 0.0
        acc = float((pred.astype(int) == self._y_test).mean())
        return f"有效提交！隐藏测试集准确率 = {acc:.4f}", acc


# ---------- 2. MockLLM：按"套路"生成越来越强的 ML 代码 ----------

CODE_TEMPLATE = '''import numpy as np
{imports}

# 契约：train.csv 带 target 列，test.csv 没有；产出 submission.csv
data = np.genfromtxt("train.csv", delimiter=",", skip_header=1)
X, y = data[:, :-1], data[:, -1]
X_test = np.genfromtxt("test.csv", delimiter=",", skip_header=1)

clf = {model}
clf.fit(X, y)
pred = clf.predict(X_test)
np.savetxt("submission.csv", pred, fmt="%d")
print("done")
'''

# 候选套路从弱到强（真 LLM 会自己想出这些；MockLLM 按固定顺序走一遍）
REPERTOIRE = [
    ("逻辑回归（线性基线）", "from sklearn.linear_model import LogisticRegression",
     "LogisticRegression(max_iter=2000)"),
    ("随机森林（100 棵）", "from sklearn.ensemble import RandomForestClassifier",
     "RandomForestClassifier(n_estimators=100, random_state=42)"),
    ("随机森林（300 棵+限深）", "from sklearn.ensemble import RandomForestClassifier",
     "RandomForestClassifier(n_estimators=300, max_depth=10, random_state=42)"),
    ("梯度提升 GBDT", "from sklearn.ensemble import GradientBoostingClassifier",
     "GradientBoostingClassifier(random_state=42)"),
]


class MockLLM:
    """仍然无状态：从 context 数"已经试了几轮"，按套路表取下一个；
    套路用完则 FINISH 并从 context 里解析出历史最高分。"""

    def __call__(self, context: str):
        scores = []
        for line in context.splitlines():
            if line.startswith("试验: ") and " -> acc=" in line:
                scores.append(float(line.rsplit("=", 1)[1]))
        idx = len(scores)
        if idx >= len(REPERTOIRE):
            best = max(scores) if scores else 0.0
            return (f"候选套路已全部试完，历史最高 acc={best:.4f}", f"FINISH: {best:.4f}")
        name, imports, model = REPERTOIRE[idx]
        thought = (f"已试 {idx} 个模型，最高 acc={max(scores, default=0):.4f}；"
                   f"本轮尝试 {name}")
        return (thought, CODE_TEMPLATE.format(imports=imports, model=model))


# ---------- 3. 提交-评分循环：本课的主角 ----------

def improve_loop(env, llm, max_rounds: int = 6):
    context = f"任务: {env.task}\n"
    print(context, end="")
    best_score, best_round = 0.0, 0
    for i in range(1, max_rounds + 1):
        thought, action = llm(context)
        if action.startswith("FINISH"):
            print(f"[第{i}轮] 思考: {thought}\n         ✅ FINISH")
            return best_score, best_round
        context += f"[第{i}轮] 思考: {thought}\n"
        obs, score = env.step(action)                       # Act + Observe（真实执行+评分）
        context += f"观察: {obs}\n试验: 第{i}轮 -> acc={score:.4f}\n"
        if score > best_score:
            best_score, best_round = score, i
        print(f"[第{i}轮] {thought}\n         动作: 提交代码 → 观察: {obs}")
    print("轮数预算耗尽")
    return best_score, best_round


if __name__ == "__main__":
    env = MLCodeEnv(dataset="breast_cancer")
    best, rnd = improve_loop(env, MockLLM(), max_rounds=6)
    print(f"\n{'='*46}\n🏆 第 {rnd} 轮取得最佳准确率 {best:.4f}"
          f"——这是隐藏测试集上的真实评分，不是 mock")
