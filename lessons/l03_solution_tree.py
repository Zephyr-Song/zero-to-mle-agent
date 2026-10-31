"""
L03 · 方案树与三算子：亲眼看到"会回头"赢"一条道走到黑"
========================================================

跑法：pip install scikit-learn  →  python lessons/l03_solution_tree.py

【本课目标】
L02 的 agent 是一条直线：试一个模型、看分、换下一个。本课把所有历史解组织成
**方案树**（AIDE 的核心结构，arXiv 2502.13138），并引入三个算子：

- DRAFT  ：从零起草一个新方案（新模型家族）
- IMPROVE：在某个已有节点上做一次原子改进（本课 = 加特征标准化）
- DEBUG  ：提交失败时修复重交（本课的模板都能跑通，所以它不会触发——
           但真实 agent 里它每天都在用，L07 接真 LLM 后你会见到）

然后做一次受控实验：**同样的预算、同样的算子、同样的代码生成器**，
只换"搜索策略 π"这一样东西，看结果差多少。

【核心公式】
AIDE 把 ML 工程形式化为：s* = argmax_{s∈S} h(s)。
方案树的每个节点 = 一个候选解 s（一段代码 + 它的真实分数 h(s)）；
一次 IMPROVE/DRAFT = 在树上长出一个孩子。

【π 与 f 的解耦（AIDE 的重要设计）】
- 搜索策略 π：决定"下一步改哪个节点、用哪个算子"
- 编码能力 f：只负责把决定变成代码
本课的两种策略共用同一个 f——所以最终差距 100% 归因于 π。

【实验设计（全部真实，无脚本安排）】
环境的训练数据里混入了一个**量纲 ×10000 的冗余特征**——就像现实数据里
混进"以分为单位的金额"。后果（你可以自己验证）：
- 树模型（rf/gbdt）对特征尺度免疫，几乎不受影响
- 距离/梯度类模型（logreg/svc/knn）被这个特征绑架，分数很难看
- 但只要"加一步标准化"，logreg 从 0.9357 → 0.9883（全场冠军）、svc → 0.9708

两种策略：
- greedy："永远只改当前最优节点；最优节点没得改了才起草新模型"
- tree  ："先广度铺开所有候选，然后**从最差的叶子开始救**（差节点可能离
           翻盘只差一步）"

【结果怎么读（诚实的 nuance）】
AIRA（L05 参考）说"算子比搜索重要"——它对比的是 MCTS vs 进化 vs greedy 这类
**都还算合理**的策略之间的差异。本课对比的是 greedy vs "会回头"：greedy 是
病态策略，它对"当前最优"的迷信让它永远错过"差节点 + 一步翻盘"的组合。
两个结论并存：回头很重要（本课），但在都会回头的策略里，算子质量才是瓶颈（AIRA）。

【思考题】
1. greedy 在 rf 上做了两次"标准化/加树"的改进，其中一次是**零增益**——
   为什么树模型加标准化没用？（这是可以写进你论文的相关工作段的事实）
2. 树策略的"worst-first"在预算只有 6 轮时会怎样？（提示：5 轮 draft 完
   只剩 1 轮 improve）。策略该不该动态调整 explore/exploit 比例？
3. 如果 IMPROVE 算子的候选只有"调超参"而没有"特征标准化"，两种策略的
   天花板分别是多少？（这就是 AIRA 说的"算子集合决定可达上限"）

【下一课预告】
L04：树越长越大，context 里该给"大脑"看什么？全堆进去会爆炸（ReAct 的死法），
只给分数又丢失线索——AIDE 的 Summarization 算子登场。
"""

import csv
import subprocess
import sys
import tempfile
import warnings
from pathlib import Path

try:
    import numpy as np
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split
except ImportError:
    raise SystemExit("本课需要 scikit-learn：先运行  pip install scikit-learn")

warnings.filterwarnings("ignore")


# ---------- 1. 环境：与 L02 同款，但数据里埋了"量纲陷阱" ----------

class MLCodeEnv:
    """训练数据混入一个 ×10000 的冗余特征（现实中=混进'以分为单位的金额'）。
    agent 看得到数据（可以自己发现量纲问题），但 MockLLM 的套路表里没有
    '删特征'这个动作——它只能靠 IMPROVE(标准化) 来翻盘。

    _CACHE：类级提交缓存——相同代码不重复执行（真实评测平台的标配：
    环境是确定性的，同一段代码必然得到同一个分数）。"""

    _CACHE: dict = {}

    def __init__(self, test_size: float = 0.3, seed: int = 42):
        data = load_breast_cancer()
        X = np.hstack([data.data, (data.data[:, 3] * 10000).reshape(-1, 1)])
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, data.target, test_size=test_size,
            random_state=seed, stratify=data.target)
        self._y_test = y_te.astype(int)
        self.workspace = Path(tempfile.mkdtemp(prefix="l03_env_"))
        self._dump(self.workspace / "train.csv",
                   [f"f{i}" for i in range(X_tr.shape[1])] + ["target"],
                   [row + [int(t)] for row, t in zip(X_tr.tolist(), y_tr)])
        self._dump(self.workspace / "test.csv",
                   [f"f{i}" for i in range(X_te.shape[1])], X_te.tolist())
        self.task = (f"二分类任务（{X_tr.shape[1]} 个特征，注意第 31 个特征量纲异常大）\n"
                     f"提交代码 → 训练 → 写 submission.csv → 环境在隐藏测试集上评分")

    @staticmethod
    def _dump(path: Path, header: list, rows: list) -> None:
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(header)
            w.writerows(rows)

    def step(self, code: str) -> tuple:
        key = hash(code)
        if key in MLCodeEnv._CACHE:  # 环境是确定性的：同代码必得同分
            return MLCodeEnv._CACHE[key]
        result = self._execute(code)
        MLCodeEnv._CACHE[key] = result
        return result

    def _execute(self, code: str) -> tuple:
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


# ---------- 2. 候选与"改进阶梯"（编码能力 f 的弹药库） ----------

# 每个家族一条改进阶梯：level 0 = DRAFT 的原始方案，往后每级 = 一次 IMPROVE
FAMILIES = {
    "rf": ("随机森林", [
        ("随机森林(100棵)", "RandomForestClassifier(n_estimators=100, random_state=42)"),
        ("随机森林+标准化", "make_pipeline(StandardScaler(), RandomForestClassifier(n_estimators=100, random_state=42))"),
        ("随机森林(300棵)", "RandomForestClassifier(n_estimators=300, random_state=42)"),
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
DRAFT_ORDER = list(FAMILIES)  # [rf, gbdt, logreg, svc, knn]

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
    """编码能力 f：把 (家族, 阶梯等级) 翻译成代码。**不含任何策略**。"""
    desc, model = FAMILIES[family][1][level]
    imports = {"rf": "from sklearn.ensemble import RandomForestClassifier",
               "gbdt": "from sklearn.ensemble import GradientBoostingClassifier",
               "logreg": "from sklearn.linear_model import LogisticRegression",
               "svc": "from sklearn.svm import SVC",
               "knn": "from sklearn.neighbors import KNeighborsClassifier"}[family]
    op = "DRAFT" if level == 0 else "IMPROVE"
    thought = (f"{op}: {desc}" if level == 0
               else f"IMPROVE: 在上一版基础上 → {desc}")
    return thought, desc, CODE_TEMPLATE.format(imports=imports, model=model)


# ---------- 3. 方案树 ----------

class Node:
    def __init__(self, nid, family, level, operator, desc, code, score, parent):
        self.id, self.family, self.level = nid, family, level
        self.operator, self.desc, self.code = operator, desc, code
        self.score, self.parent = score, parent
        self.children: list = []


class SolutionTree:
    def __init__(self):
        self.nodes: dict = {}
        self._n = 0

    def add(self, family, level, operator, desc, code, score, parent=None) -> str:
        self._n += 1
        nid = f"n{self._n}"
        self.nodes[nid] = Node(nid, family, level, operator, desc, code, score, parent)
        if parent:
            self.nodes[parent].children.append(nid)
        return nid

    def best(self) -> Node:
        return max(self.nodes.values(), key=lambda n: n.score) if self.nodes else None

    def has_room(self, node: Node) -> bool:
        """阶梯还有下一级，且这一级没被试过（防止对同一节点重复同一改进）。"""
        nxt = node.level + 1
        if nxt >= len(FAMILIES[node.family][1]):
            return False
        return not any(self.nodes[c].level == nxt for c in node.children)

    def worst_with_room(self) -> Node:
        """还有改进空间的节点里分数最低的那个（树策略的'回头'对象）。"""
        room = [n for n in self.nodes.values() if self.has_room(n)]
        return min(room, key=lambda n: n.score) if room else None

    def render(self):
        print("  方案树：")
        roots = [n for n in self.nodes.values() if n.parent is None]
        best_id = self.best().id

        def walk(nid, prefix):
            n = self.nodes[nid]
            star = "  ⭐全场最优" if nid == best_id else ""
            print(f"{prefix}[{nid}] {n.operator:7s} {n.desc:14s} acc={n.score:.4f}{star}")
            for c in n.children:
                walk(c, prefix + "     └─ ")

        for r in roots:
            walk(r.id, "  ")


# ---------- 4. 两种搜索策略 π（唯一变量！） ----------

def greedy_policy(tree: SolutionTree):
    """永远只改当前最优节点；最优没得改了才起草新模型，从不回头看差节点。"""
    best = tree.best()
    if best is not None and tree.has_room(best):
        return ("IMPROVE", best)
    for fam in DRAFT_ORDER:
        if not any(n.family == fam and n.level == 0 for n in tree.nodes.values()):
            return ("DRAFT", fam)
    return None


def tree_policy(tree: SolutionTree):
    """先广度铺开所有候选，然后从最差的节点开始救（worst-first 回头）。"""
    for fam in DRAFT_ORDER:
        if not any(n.family == fam and n.level == 0 for n in tree.nodes.values()):
            return ("DRAFT", fam)
    leaf = tree.worst_with_room()
    return ("IMPROVE", leaf) if leaf else None


# ---------- 5. 受控实验：同预算、同算子、同生成器，只换 π ----------

def run(policy, budget: int, label: str):
    print(f"\n{'='*56}\n策略 = {label}（预算 {budget} 轮）\n{'='*56}")
    env, tree = MLCodeEnv(), SolutionTree()
    wasted = 0
    for r in range(1, budget + 1):
        sel = policy(tree)
        if sel is None:
            wasted += 1
            print(f"  [r{r}] ⏸ 策略找不到可做的事——预算被浪费")
            continue
        op, target = sel
        if op == "DRAFT":
            thought, desc, code = generate(target, 0)
            parent, family, level = None, target, 0
        else:
            thought, desc, code = generate(target.family, target.level + 1)
            parent, family, level = target.id, target.family, target.level + 1
        obs, score = env.step(code)
        tree.add(family, level, op, desc, code, score, parent)
        print(f"  [r{r}] {thought} → acc={score:.4f}")
    tree.render()
    print(f"  空转/浪费轮数: {wasted} · 最优: {tree.best().desc} = {tree.best().score:.4f}")
    return tree.best().score


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "both"  # greedy | tree | both
    BUDGET = 8
    g = t = None
    if which in ("greedy", "both"):
        g = run(greedy_policy, BUDGET, "greedy：只改当前最优节点")
    if which in ("tree", "both"):
        t = run(tree_policy, BUDGET, "tree：广度铺开 + worst-first 回头")
    if which == "both":
        print(f"\n{'='*56}\n🏁 同预算({BUDGET})、同算子、同代码生成器，只换搜索策略：")
        print(f"   greedy = {g:.4f}   vs   tree = {t:.4f}")
        print("   差距全部来自'敢不敢回头救看起来没救的节点'——")
        print("   greedy 把 0.93 的逻辑回归判了死刑，却不知道它离全场冠军只差一步标准化。")
