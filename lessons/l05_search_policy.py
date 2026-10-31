"""
L05 · 搜索策略的上限：亲手验证 AIRA 的"算子才是瓶颈"
======================================================

跑法：pip install scikit-learn  →  python lessons/l05_search_policy.py

【本课要回答的问题】
改进一个 agent，钱该花在哪？换更聪明的**搜索策略**（greedy→MCTS→进化），
还是换更强的**算子集合**（给 agent 更多/更好的可用动作）？

AIRA（Meta, NeurIPS 2025, arXiv 2507.02554）的答案是"算子"：它保留 AIDE
的算子、只换搜索策略，发现 MCTS / 进化相对 greedy 增益极小，甚至关掉
AIDE 的 Memory 算子奖牌率几乎不变。

【本课实验：2 × 4 受控矩阵】
- 算子集合两档：
  - strong：IMPROVE 能"加特征标准化"（救活被量纲绑架的模型）
  - weak  ：IMPROVE 只能"调超参"（改树的棵数 / C / k / 学习率）
- 搜索策略四种：greedy（只改最优）/ worst_first（L03 的回头）
  / uct（MCTS 式：分数 + 探索奖励）/ evolution（精英变异）
- 同预算（8 轮）、同环境、同代码生成器 → 只看最终最优分。

【你会看到什么（诚实预告）】
1. 强算子下 greedy 明显吃亏（它在 L03 就输过），但**三种"会回头"的策略
   彼此差异很小**——这就是 AIRA 说的"搜索算法差异不大"的适用范围。
2. 弱算子下，四种策略**全部被同一个天花板锁死**——换什么搜索都没用。
3. 于是边界清楚了：
   **算子集合决定"可达上限"，搜索策略只决定"能不能摸到那个上限"。**
   AIRA 的结论成立的前提是"搜索策略已经都会回头了"。

【思考题】
1. 如果给 weak 算子集再加一个"删掉量纲异常特征"的算子，天花板会变吗？
   （提示：这才是真正对症的算子——想想它能到多少分。）
2. UCT 的探索奖励在本课几乎没起作用，为什么？（提示：候选空间只有 5 个家族、
   预算只有 8 轮——探索奖励要在多大的空间里才值回票价？）
3. 把这个结论翻译成你自己的研究：如果你想让 MLE agent 更强，应该先加
   什么样的算子？怎么证明它真的扩大了可达空间？

【下一课预告】
L06：给 agent 装一道**人类审计闸门**（本系列的原创创新点）——
它不只关心分数，还关心"这个分数是不是刷出来的"。
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


# ---------- 1. 环境（与 L03/L04 相同：量纲陷阱 + 真实评分 + 缓存） ----------

class MLCodeEnv:
    _CACHE: dict = {}

    def __init__(self, test_size: float = 0.3, seed: int = 42):
        data = load_breast_cancer()
        X = np.hstack([data.data, (data.data[:, 3] * 10000).reshape(-1, 1)])
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, data.target, test_size=test_size,
            random_state=seed, stratify=data.target)
        self._y_test = y_te.astype(int)
        self.workspace = Path(tempfile.mkdtemp(prefix="l05_env_"))
        self._dump(self.workspace / "train.csv",
                   [f"f{i}" for i in range(X_tr.shape[1])] + ["target"],
                   [row + [int(t)] for row, t in zip(X_tr.tolist(), y_tr)])
        self._dump(self.workspace / "test.csv",
                   [f"f{i}" for i in range(X_te.shape[1])], X_te.tolist())
        self.task = f"二分类任务（{X_tr.shape[1]} 个特征，最后一个量纲异常大）"

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
            return "子进程超时", 0.0
        sub = self.workspace / "submission.csv"
        if not sub.exists():
            return "没有产出 submission.csv", 0.0
        pred = np.loadtxt(sub, delimiter=",")
        if pred.shape[0] != len(self._y_test):
            return "预测数量不符", 0.0
        return "有效提交", float((pred.astype(int) == self._y_test).mean())


# ---------- 2. 两档算子集合（本课的第一个自变量） ----------

S = "make_pipeline(StandardScaler(), {})"
OPERATOR_SETS = {
    "strong": {  # IMPROVE 能"加标准化"——对症，能救活被量纲绑架的模型
        "rf": [("随机森林(100棵)", "RandomForestClassifier(n_estimators=100, random_state=42)"),
               ("随机森林+标准化", S.format("RandomForestClassifier(n_estimators=100, random_state=42)"))],
        "gbdt": [("GBDT", "GradientBoostingClassifier(random_state=42)"),
                 ("GBDT+标准化", S.format("GradientBoostingClassifier(random_state=42)"))],
        "logreg": [("逻辑回归", "LogisticRegression(max_iter=2000)"),
                   ("逻辑回归+标准化", S.format("LogisticRegression(max_iter=2000)"))],
        "svc": [("SVM(RBF)", "SVC(random_state=42)"),
                ("SVM+标准化", S.format("SVC(random_state=42)"))],
        "knn": [("KNN(k=5)", "KNeighborsClassifier(n_neighbors=5)"),
                ("KNN+标准化", S.format("KNeighborsClassifier(n_neighbors=5)"))],
    },
    "weak": {  # IMPROVE 只能"调超参"——不对症，量纲问题永远解决不了
        "rf": [("随机森林(100棵)", "RandomForestClassifier(n_estimators=100, random_state=42)"),
               ("随机森林(200棵)", "RandomForestClassifier(n_estimators=200, random_state=42)")],
        "gbdt": [("GBDT(lr=0.1)", "GradientBoostingClassifier(random_state=42)"),
                 ("GBDT(lr=0.05,n=200)", "GradientBoostingClassifier(learning_rate=0.05, n_estimators=200, random_state=42)")],
        "logreg": [("逻辑回归(C=1)", "LogisticRegression(max_iter=2000)"),
                   ("逻辑回归(C=0.1)", "LogisticRegression(max_iter=2000, C=0.1)")],
        "svc": [("SVM(C=1)", "SVC(random_state=42)"),
                ("SVM(C=0.5)", "SVC(random_state=42, C=0.5)")],
        "knn": [("KNN(k=5)", "KNeighborsClassifier(n_neighbors=5)"),
                ("KNN(k=3)", "KNeighborsClassifier(n_neighbors=3)")],
    },
}
FAMILY_IMPORTS = {"rf": "from sklearn.ensemble import RandomForestClassifier",
                  "gbdt": "from sklearn.ensemble import GradientBoostingClassifier",
                  "logreg": "from sklearn.linear_model import LogisticRegression",
                  "svc": "from sklearn.svm import SVC",
                  "knn": "from sklearn.neighbors import KNeighborsClassifier"}
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


# ---------- 3. 方案树 ----------

class Node:
    def __init__(self, nid, family, level, desc, score, parent, code):
        self.id, self.family, self.level = nid, family, level
        self.desc, self.score, self.parent, self.code = desc, score, parent, code
        self.visits = 1  # 被"选中/扩展"的次数，UCT 要用的计数器
        self.children: list = []

    def ladder(self, opset):
        return OPERATOR_SETS[opset][self.family]


class SolutionTree:
    def __init__(self, opset: str):
        self.opset, self.nodes, self._n = opset, {}, 0

    def add(self, family, level, desc, score, parent, code) -> str:
        self._n += 1
        nid = f"n{self._n}"
        self.nodes[nid] = Node(nid, family, level, desc, score, parent, code)
        if parent:
            self.nodes[parent].children.append(nid)
        return nid

    def best(self):
        return max(self.nodes.values(), key=lambda n: n.score) if self.nodes else None

    def has_room(self, node) -> bool:
        nxt = node.level + 1
        if nxt >= len(node.ladder(self.opset)):
            return False
        return not any(self.nodes[c].level == nxt for c in node.children)

    def room(self):
        return [n for n in self.nodes.values() if self.has_room(n)]

    def drafted(self, fam):
        return any(n.family == fam and n.level == 0 for n in self.nodes.values())


def generate(opset, family, level):
    desc, model = OPERATOR_SETS[opset][family][level]
    return desc, CODE_TEMPLATE.format(imports=FAMILY_IMPORTS[family], model=model)


# ---------- 4. 四种搜索策略（本课的第二个自变量） ----------

def _draft_first(tree):
    for fam in OPERATOR_SETS[tree.opset]:
        if not tree.drafted(fam):
            return ("DRAFT", fam)
    return None


def greedy_policy(tree):
    """只改当前最优节点；最优没得改了才起草新模型。"""
    best = tree.best()
    if best is not None and tree.has_room(best):
        return ("IMPROVE", best)
    return _draft_first(tree)


def worst_first_policy(tree):
    """L03 的策略：广度铺开 + 从最差的节点开始救。"""
    d = _draft_first(tree)
    if d:
        return d
    room = tree.room()
    if not room:
        return None
    return ("IMPROVE", min(room, key=lambda n: n.score))


def uct_policy(tree, c: float = 1.0):
    """MCTS 式：分数 + 探索奖励 c·sqrt(ln N / visits)。"""
    d = _draft_first(tree)
    if d:
        return d
    room = tree.room()
    if not room:
        return None
    n_total = sum(n.visits for n in tree.nodes.values())
    def bonus(n):
        return n.score + c * np.sqrt(np.log(max(n_total, 2)) / n.visits)
    return ("IMPROVE", max(room, key=bonus))


def evolution_policy(tree, elite: int = 2):
    """进化式：保留前 elite 个精英，轮流对它们做变异（改进）。"""
    d = _draft_first(tree)
    if d:
        return d
    room = tree.room()
    if not room:
        return None
    elites = sorted(room, key=lambda n: -n.score)[:elite]
    return ("IMPROVE", min(elites, key=lambda n: n.visits))


# ---------- 5. 2 × 4 受控矩阵 ----------

POLICIES = {"greedy": greedy_policy, "worst_first": worst_first_policy,
            "uct": uct_policy, "evolution": evolution_policy}


def run(opset: str, policy, budget: int = 8):
    env, tree = MLCodeEnv(), SolutionTree(opset)
    for _ in range(budget):
        sel = policy(tree)
        if sel is None:
            break
        op, target = sel
        if op == "DRAFT":
            desc, code = generate(opset, target, 0)
            parent, family, level = None, target, 0
        else:
            target.visits += 1
            desc, code = generate(opset, target.family, target.level + 1)
            parent, family, level = target.id, target.family, target.level + 1
        _, score = env.step(code)
        tree.add(family, level, desc, score, parent, code)
    return tree.best().score, tree.best().desc


if __name__ == "__main__":
    BUDGET = 8
    results = {}
    print(f"2×4 矩阵：算子集合 {{strong, weak}} × 搜索策略 {{greedy, worst_first, uct, evolution}}"
          f"（预算 {BUDGET} 轮）\n")
    for opset in ("strong", "weak"):
        print(f"  ── 算子集合 = {opset} "
              f"({'能加标准化' if opset == 'strong' else '只能调超参'}) ──")
        for name, fn in POLICIES.items():
            s, d = run(opset, fn, BUDGET)
            results[(opset, name)] = s
            print(f"    {name:<12s} 最优 = {s:.4f}   ({d})")
        print()

    strong = [results[("strong", n)] for n in POLICIES]
    weak = [results[("weak", n)] for n in POLICIES]
    backtracking = [results[("strong", n)] for n in ("worst_first", "uct", "evolution")]
    print("  结论（全部实测）：")
    print(f"    · 算子效应：最好的策略下 strong-weak = "
          f"{max(strong) - max(weak):+.4f}  ← 算子把天花板抬高了这么多")
    print(f"    · 搜索效应：strong 算子下 最好-最差 = "
          f"{max(strong) - min(strong):+.4f}（其中 greedy 是病态策略）")
    print(f"    · 三种'会回头'策略彼此差异 = {max(backtracking) - min(backtracking):.4f}"
          f"  ← 很小，这就是 AIRA 的适用范围")
    print("    · 一句话：算子决定可达上限，搜索只决定你能不能摸到那个上限。")
