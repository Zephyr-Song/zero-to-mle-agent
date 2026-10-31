"""
L04 · Summarization：上下文不爆炸的关键（AIDE 碾压 ReAct 的真正原因）
======================================================================

跑法：pip install scikit-learn  →  python lessons/l04_summarization.py

【本课要回答的问题】
L03 的树越长越大，而 LLM 是**无状态**的——所谓"记忆"，就是每一轮把历史
重新拼进 prompt 再发一次。那么该发什么？

- **ReAct 式（naive）**：把所有历史全堆进去——包括**每一版提交过的完整代码**。
  （agent 要"在上一版基础上改"，于是把历史代码全带上，看起来很合理？）
- **AIDE 的 Σ（Summarization）**：每个节点只留一行摘要（配置 + 分数），
  改哪个节点就只带**那一份**代码。无状态视角，不堆历史。

这正是 AIDE 论文批评 ReAct 的原话："continually appending all historical
data can lead to oversized prompts"——historical data 的主体就是历史代码。

【本课实验（数字全部真实测量）】
跑一次和 L03 完全相同的树搜索（同环境、同策略、同预算），逐轮测量两种
记账方式下"这一轮要发给 LLM 的上下文大小"和"累计发送量"（=真实 API 成本）。

【你会看到的核心事实】
1. 两种方式做出的**决策完全一致**（最优解同一个节点、同一个分数）——
   Σ 省下的不是搜索质量，是**数量级的成本**。
2. 每轮发送量：ReAct 线性增长，Σ 近似常数；累计成本 O(n²) vs O(n)。
3. 真实 LLM 上还有第二重收益：上下文越长，"lost in the middle"越严重，
   决策质量反而下降——省钱又提质。

【思考题】
1. Σ 为什么只保留"无状态"信息（分数/配置），而丢弃"第几轮试的"这种时序？
   （提示：回想 AIDE 的公式 s* = argmax h(s)——目标是无状态的。）
2. Σ 每轮只带"被改节点的那一份代码"。如果 IMPROVE 要同时看两个节点
   （比如 AIRA 的 Crossover 算子），成本会怎么变？还能更省吗？
3. 看课程结尾的外推：预算 8→50 轮，成本比从 ~4× 拉到 ~10×。算一算
   50→200 轮会怎样？这就是长程 agent 必须做 Σ 的数学原因。

【下一课预告】
L05：搜索策略的上限在哪里？greedy / MCTS / 进化搜索摆上桌，
亲手验证 AIRA 的"算子才是瓶颈"到底在什么范围内成立。
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


# ---------- 1. 环境：与 L03 相同（量纲陷阱 + 真实评分 + 提交缓存） ----------

class MLCodeEnv:
    _CACHE: dict = {}  # 确定性环境：同代码必得同分，不重复执行

    def __init__(self, test_size: float = 0.3, seed: int = 42):
        data = load_breast_cancer()
        X = np.hstack([data.data, (data.data[:, 3] * 10000).reshape(-1, 1)])
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, data.target, test_size=test_size,
            random_state=seed, stratify=data.target)
        self._y_test = y_te.astype(int)
        self.workspace = Path(tempfile.mkdtemp(prefix="l04_env_"))
        self._dump(self.workspace / "train.csv",
                   [f"f{i}" for i in range(X_tr.shape[1])] + ["target"],
                   [row + [int(t)] for row, t in zip(X_tr.tolist(), y_tr)])
        self._dump(self.workspace / "test.csv",
                   [f"f{i}" for i in range(X_te.shape[1])], X_te.tolist())
        self.task = (f"二分类任务（{X_tr.shape[1]} 个特征，注意最后一个特征量纲异常大）"
                     f"提交代码→训练→写 submission.csv→隐藏测试集评分")

    @staticmethod
    def _dump(path, header, rows):
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(header)
            w.writerows(rows)

    def step(self, code: str):
        key = hash(code)
        if key in MLCodeEnv._CACHE:
            return MLCodeEnv._CACHE[key]
        result = self._execute(code)
        MLCodeEnv._CACHE[key] = result
        return result

    def _execute(self, code: str):
        (self.workspace / "solution.py").write_text(code, encoding="utf-8")
        try:
            subprocess.run([sys.executable, "solution.py"], cwd=self.workspace,
                           timeout=120, capture_output=True, text=True)
        except subprocess.TimeoutExpired:
            return "子进程超时——无效提交", 0.0
        sub = self.workspace / "submission.csv"
        if not sub.exists():
            return "没有产出 submission.csv——无效提交", 0.0
        pred = np.loadtxt(sub, delimiter=",")
        if pred.shape[0] != len(self._y_test):
            return "预测数量不符——无效提交", 0.0
        acc = float((pred.astype(int) == self._y_test).mean())
        return f"有效提交，acc={acc:.4f}", acc


# ---------- 2. 候选、改进阶梯、代码生成（与 L03 相同） ----------

FAMILIES = {
    "rf": ("随机森林", [
        ("随机森林(100棵)", "RandomForestClassifier(n_estimators=100, random_state=42)"),
        ("随机森林+标准化", "make_pipeline(StandardScaler(), RandomForestClassifier(n_estimators=100, random_state=42))"),
    ]),
    "gbdt": ("梯度提升", [
        ("GBDT", "GradientBoostingClassifier(random_state=42)"),
        ("GBDT+标准化", "make_pipeline(StandardScaler(), GradientBoostingClassifier(random_state=42))"),
    ]),
    "logreg": ("逻辑回归", [
        ("逻辑回归", "LogisticRegression(max_iter=2000)"),
        ("逻辑回归+标准化", "make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))"),
    ]),
    "svc": ("SVM(RBF)", [
        ("SVM(RBF)", "SVC(random_state=42)"),
        ("SVM+标准化", "make_pipeline(StandardScaler(), SVC(random_state=42))"),
    ]),
    "knn": ("K近邻", [
        ("KNN(k=5)", "KNeighborsClassifier(n_neighbors=5)"),
        ("KNN+标准化", "make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5))"),
    ]),
}
DRAFT_ORDER = list(FAMILIES)

CODE_TEMPLATE = '''import numpy as np
{imports}
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

data = np.genfromtxt("train.csv", delimiter=",", skip_header=1)
X, y = data[:, :-1], data[:, -1]
X_test = np.genfromtxt("test.csv", delimiter=",", skip_header=1)

clf = {model}
clf.fit(X, y)
pred = clf.predict(X_test)
np.savetxt("submission.csv", pred, fmt="%d")
print("done")
'''


def generate(family: str, level: int):
    desc, model = FAMILIES[family][1][level]
    imports = {"rf": "from sklearn.ensemble import RandomForestClassifier",
               "gbdt": "from sklearn.ensemble import GradientBoostingClassifier",
               "logreg": "from sklearn.linear_model import LogisticRegression",
               "svc": "from sklearn.svm import SVC",
               "knn": "from sklearn.neighbors import KNeighborsClassifier"}[family]
    op = "DRAFT" if level == 0 else "IMPROVE"
    thought = f"{op}: {desc}"
    return thought, desc, CODE_TEMPLATE.format(imports=imports, model=model)


# ---------- 3. 方案树（与 L03 相同 + Σ 摘要） ----------

class Node:
    def __init__(self, nid, family, level, operator, desc, score, parent, code):
        self.id, self.family, self.level = nid, family, level
        self.operator, self.desc, self.score = operator, desc, score
        self.parent, self.code = parent, code
        self.children: list = []


class SolutionTree:
    def __init__(self):
        self.nodes: dict = {}
        self._n = 0

    def add(self, family, level, operator, desc, score, parent, code) -> str:
        self._n += 1
        nid = f"n{self._n}"
        self.nodes[nid] = Node(nid, family, level, operator, desc, score, parent, code)
        if parent:
            self.nodes[parent].children.append(nid)
        return nid

    def best(self):
        return max(self.nodes.values(), key=lambda n: n.score) if self.nodes else None

    def has_room(self, node: Node) -> bool:
        nxt = node.level + 1
        if nxt >= len(FAMILIES[node.family][1]):
            return False
        return not any(self.nodes[c].level == nxt for c in node.children)

    def worst_with_room(self):
        room = [n for n in self.nodes.values() if self.has_room(n)]
        return min(room, key=lambda n: n.score) if room else None

    def summary_lines(self) -> list:
        """Σ 的核心：每个节点一行——配置 + 分数，没有时序、没有历史代码。"""
        return [f"[{n.id}] {n.desc} acc={n.score:.4f}" for n in self.nodes.values()]


def tree_policy(tree: SolutionTree):
    for fam in DRAFT_ORDER:
        if not any(n.family == fam and n.level == 0 for n in tree.nodes.values()):
            return ("DRAFT", fam)
    leaf = tree.worst_with_room()
    return ("IMPROVE", leaf) if leaf else None


# ---------- 4. 本课主角：两种上下文记账 ----------

def run_with_accounting(budget: int = 8):
    env, tree = MLCodeEnv(), SolutionTree()
    task_len = len(env.task)
    naive_transcript = ""           # ReAct：全量历史（含每版完整代码）
    naive_sent, aide_sent = [], []  # 每轮"要发给 LLM"的字符数

    for r in range(1, budget + 1):
        op, target = tree_policy(tree)
        if op == "DRAFT":
            thought, desc, code = generate(target, 0)
            parent, family, level = None, target, 0
            code_needed = 0  # 起草新方案不需要旧代码
        else:
            thought, desc, code = generate(target.family, target.level + 1)
            parent, family, level = target.id, target.family, target.level + 1
            code_needed = len(target.code)  # Σ 只带被改节点这一份代码
        obs, score = env.step(code)
        tree.add(family, level, op, desc, score, parent, code)
        target_node = tree.nodes[parent] if parent else None

        # ReAct 记账：历史里堆着每一版完整代码，且每轮全部重发
        naive_transcript += f"{thought}\n{obs}\n{code}\n"
        naive_sent.append(task_len + len(naive_transcript))
        # Σ 记账：任务 + 每节点一行摘要 + 仅当前要改的那一份代码
        aide_turn = task_len + sum(len(s) for s in tree.summary_lines()) + code_needed
        aide_sent.append(aide_turn)

        print(f"  [r{r}] {thought} → acc={score:.4f}   "
              f"本轮发送: ReAct={naive_sent[-1]:,} vs Σ={aide_sent[-1]:,}")

    # ---- 累计成本（这就是真实 API 账单的形状）----
    tn, ta = sum(naive_sent), sum(aide_sent)
    print(f"\n  累计发送量对比（LLM 每轮重发整个上下文 → 真实 API 成本）：")
    bar_n = max(1, int(tn / max(tn, ta) * 34))
    bar_a = max(1, int(ta / max(tn, ta) * 34))
    print(f"  ReAct全量 {'█' * bar_n} {tn:,} 字符 ≈ {tn // 4:,} tokens")
    print(f"  AIDE Σ    {'█' * bar_a} {ta:,} 字符 ≈ {ta // 4:,} tokens")
    print(f"  → 成本比 {tn / ta:.1f}×，且决策完全一致：最优都是 "
          f"{tree.best().desc} = {tree.best().score:.4f}")

    # ---- 外推：预算加大后差距怎么拉开（用实测均值做诚实外推）----
    avg_code = sum(len(n.code) for n in tree.nodes.values()) / len(tree.nodes)
    avg_summary = sum(len(s) for s in tree.summary_lines()) / len(tree.nodes)

    def extrapolate(n_rounds):
        # naive: 每轮重发 历史(≈avg_code+obs)×轮数；Σ: 摘要×轮数 + 一份代码
        naive = sum(task_len + i * (avg_code + 60) for i in range(1, n_rounds + 1))
        aide = sum(task_len + i * avg_summary + avg_code for i in range(1, n_rounds + 1))
        return naive, aide

    for n_rounds in (8, 50, 200):
        nv, ad = extrapolate(n_rounds)
        print(f"  外推预算 {n_rounds:>3} 轮: ReAct ≈ {nv:>9,.0f} 字符 vs Σ ≈ {ad:>8,.0f}"
              f"   → 成本比 {nv / ad:.1f}×")


if __name__ == "__main__":
    run_with_accounting(budget=8)
