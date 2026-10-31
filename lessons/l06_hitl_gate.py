"""
L06 · HITL 闸门：给 agent 加人类审计层（本系列原创创新点）
=========================================================

跑法：pip install scikit-learn  →  python lessons/l06_hitl_gate.py

【本课要回答的问题】
MLGym（Meta, arXiv 2502.14499）把 agent 的自主性分成 L0–L5，关键边界在 **L1→L2**：
- L1 = "人类在环"：agent 自己干活，但高风险动作要人点头；
- L2 = "人类离环"：agent 自己拍板，人不在回路里。

我们前面造的 agent（L02–L05）都还在 L1 以下：它自己写代码、自己选模型、自己决定
ship 哪一个。问题来了——它 ship 的判定标准是什么？是"验证集分数最高"。但验证集分数
是 agent 亲眼盯着优化出来的：它既是运动员，又是裁判。一旦验证集被量纲陷阱 / 噪声 /
数据泄漏绑架，agent 会**极其自信地 ship 一个"在验证集很好、在真实数据很差"的模型**。

本课亲手造一道 HITL（Human-In-The-Loop）闸门，卡在 L1→L2 之间：
- 它不让 agent 直接 ship；
- 它用一道 agent 从没优化过、也从没见过的"审计信号"（held-out 测试集）重新打分；
- 如果两道信号打架（验证集冠军 ≠ 测试集冠军），闸门**拦截**，把决定权交还人类。

这正是我们研究仓 mle-agent-hitl 里的 src/oversight/hitl_gate.py 的"教学最小版"——
把一道可信闸门楔进 MLGym 的 L1→L2 之间，就是这条路线的原创切入点。

【本课实验：同环境，两种自主性级别】
- 环境：和 L05 一样有"量纲异常特征"（某特征 ×10000），但这次拆成 训练 / 验证 / 测试
  三份。agent 搜索时**只能看到验证集**（运动员视角）；测试集只在闸门审计时露面
  （裁判视角，agent 从没见过——代码里由约定强制：agent 只能调 val_score）。
- 算子集：复用 L05 的 weak（只能调超参，治不了量纲）和 strong（能加标准化）。
- 两个被试：
  · L1 agent：选验证集冠军，直接 ship（人类"在环"但闸没接上 = 实际离环）。
  · L2 agent：选验证集冠军 → 闸门审计测试集 → 两信号打架则拦截、交还人类选测试冠军。

【你会看到什么（诚实预告，全部实测、固定种子可复现）】
1. weak 场景：L1 ship 了 RF100（验证 0.9718）。闸门审计发现它在测试集只有 0.9510，
   而真正的测试冠军是 GBDT（0.9580）。闸门【拦截】，把"该 ship GBDT"交还人类。
   L1 会蒙着头 ship 一个次优模型，L2 拦住了。
2. strong 场景：L1 和 L2 选的是同一个模型（LR_C1+S，测试 0.9650），闸门【放行】。
   说明闸门不是"为了拦而拦"——验证集可信时它不捣乱，校准良好。

【思考题】
1. 闸门现在只看"冠军易主 + 分数落差"。要更聪明还能加什么信号？
   （提示：train vs val 落差 = 过拟合；那个 ×10000 特征权重异常大 = 泄漏嫌疑）
2. 这道闸门能不能用"真·人类"替代测试集？什么情况必须用真人类，什么情况一个独立
   指标就够？（提示：当"正确"无法用指标定义时——伦理、公平、可解释性）
3. 翻译成你的研究：要写 CHI/SIGCHI 论文，闸门拦下的那一票不该是"模型更准"，
   而该是"模型更可信 / 更负责任"。你的闸门该审计什么？

【下一课预告】
L07：把 MockLLM 换成真实的 OpenAI 兼容接口，让 agent 真的"思考"而不是读剧本，
跑一个真实小基准。
"""

import os
import csv
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

try:
    import numpy as np  # noqa
    from sklearn.datasets import load_breast_cancer  # noqa
    from sklearn.model_selection import train_test_split  # noqa
except ImportError:
    raise SystemExit("本课需要 scikit-learn：先运行  pip install scikit-learn")


# ---------- 1. 环境：训练 / 验证 / 测试 三份，测试集是"裁判" ----------

PIPE = "make_pipeline(StandardScaler(), {})"
IMPORTS = {
    "rf": "from sklearn.ensemble import RandomForestClassifier",
    "gbdt": "from sklearn.ensemble import GradientBoostingClassifier",
    "logreg": "from sklearn.linear_model import LogisticRegression",
    "svc": "from sklearn.svm import SVC",
    "knn": "from sklearn.neighbors import KNeighborsClassifier",
}
# 每个算子集合 = 一个"候选阶梯"。agent 会从这里挑它眼中的最优。
OPERATOR_SETS = {
    # weak：只能调超参，量纲问题永远治不了 → 验证集容易被噪声绑架
    "weak": [
        ("RF100", "rf", "RandomForestClassifier(n_estimators=100, random_state=42)"),
        ("RF200", "rf", "RandomForestClassifier(n_estimators=200, random_state=42)"),
        ("GBDT", "gbdt", "GradientBoostingClassifier(random_state=42)"),
        ("GBDT2", "gbdt", "GradientBoostingClassifier(learning_rate=0.05, n_estimators=200, random_state=42)"),
        ("LR_C1", "logreg", "LogisticRegression(max_iter=2000)"),
        ("LR_C01", "logreg", "LogisticRegression(max_iter=2000, C=0.1)"),
        ("SVC1", "svc", "SVC(random_state=42)"),
        ("SVC05", "svc", "SVC(random_state=42, C=0.5)"),
        ("KNN5", "knn", "KNeighborsClassifier(n_neighbors=5)"),
        ("KNN3", "knn", "KNeighborsClassifier(n_neighbors=3)"),
    ],
    # strong：能给模型"加标准化" → 量纲问题被对症解决，验证集恢复可信
    "strong": [
        ("RF100", "rf", "RandomForestClassifier(n_estimators=100, random_state=42)"),
        ("RF100+S", "rf", PIPE.format("RandomForestClassifier(n_estimators=100, random_state=42)")),
        ("GBDT", "gbdt", "GradientBoostingClassifier(random_state=42)"),
        ("GBDT+S", "gbdt", PIPE.format("GradientBoostingClassifier(random_state=42)")),
        ("LR_C1", "logreg", "LogisticRegression(max_iter=2000)"),
        ("LR_C1+S", "logreg", PIPE.format("LogisticRegression(max_iter=2000)")),
        ("SVC1", "svc", "SVC(random_state=42)"),
        ("SVC1+S", "svc", PIPE.format("SVC(random_state=42)")),
        ("KNN5", "knn", "KNeighborsClassifier(n_neighbors=5)"),
        ("KNN5+S", "knn", PIPE.format("KNeighborsClassifier(n_neighbors=5)")),
    ],
}
CODE_TEMPLATE = '''import numpy as np
{imports}
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
data = np.genfromtxt("train.csv", delimiter=",", skip_header=1)
X, y = data[:, :-1], data[:, -1]
X_test = np.genfromtxt("test.csv", delimiter=",", skip_header=1)
clf = {model}
clf.fit(X, y)
np.savetxt("submission.csv", clf.predict(X_test), fmt="%d")
print("done")
'''


class MLCodeEnv:
    """和 L02–L05 一样：子进程隔离执行 agent 写的代码（穷人版 Docker）。

    关键约定（信息不对称的建模）：
    - `val_score(code)`：agent 允许调用的"运动员指标"——它在搜索时盯着这个优化。
    - `test_score(code)`：只有**闸门**允许调用——agent 从没见过这套数据。
    这个约定就是"人类在环"的本质：agent 看不到审计用的那把尺子。
    """

    def __init__(self, opset: str, seed: int = 1):
        self.opset = opset
        self._cache: dict = {}
        data = load_breast_cancer()
        # 注入 L05 同款量纲陷阱：第 4 个特征 ×10000
        X = np.hstack([data.data, (data.data[:, 3] * 10000).reshape(-1, 1)])
        y = data.target
        # 训练 50%，剩下 50% 再对半拆成 验证 / 测试（固定种子 → 可复现）
        X_tr, X_tmp, y_tr, y_tmp = train_test_split(
            X, y, test_size=0.5, random_state=seed, stratify=y)
        X_val, X_te, y_val, y_te = train_test_split(
            X_tmp, y_tmp, test_size=0.5, random_state=seed, stratify=y_tmp)
        self._y_val = y_val.astype(int)
        self._y_test = y_te.astype(int)
        self.ws = Path(tempfile.mkdtemp(prefix="l06_env_"))
        self._dump("train.csv", [f"f{i}" for i in range(X_tr.shape[1])] + ["target"],
                   [r + [int(t)] for r, t in zip(X_tr.tolist(), y_tr)])
        self._dump("val.csv", [f"f{i}" for i in range(X_val.shape[1])], X_val.tolist())
        self._dump("test.csv", [f"f{i}" for i in range(X_te.shape[1])], X_te.tolist())

    def _dump(self, name, header, rows):
        with open(self.ws / name, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(header)
            w.writerows(rows)

    def _score(self, code: str, split: str) -> float:
        key = (hash(code), split)
        if key in self._cache:
            return self._cache[key]
        swapped = False
        try:
            if split == "val":
                # 把验证集临时顶替成 agent 眼里的 "test.csv"
                os.replace(self.ws / "test.csv", self.ws / "_hold.csv")
                os.replace(self.ws / "val.csv", self.ws / "test.csv")
                swapped = True
            (self.ws / "solution.py").write_text(code, encoding="utf-8")
            subprocess.run([sys.executable, "solution.py"], cwd=self.ws,
                           timeout=120, capture_output=True, text=True)
            sub = self.ws / "submission.csv"
            if not sub.exists():
                return 0.0
            pred = np.loadtxt(sub, delimiter=",")
            yt = self._y_val if split == "val" else self._y_test
            if pred.shape[0] != len(yt):
                return 0.0
            return float((pred.astype(int) == yt).mean())
        finally:
            if swapped:
                os.replace(self.ws / "test.csv", self.ws / "val.csv")
                os.replace(self.ws / "_hold.csv", self.ws / "test.csv")

    def _code(self, family, model):
        return CODE_TEMPLATE.format(imports=IMPORTS[family], model=model)

    # —— agent 只能调这个 ——
    def val_score(self, desc: str) -> float:
        fam, model = self._lookup(desc)
        return self._score(self._code(fam, model), "val")

    # —— 只有闸门能调这个 ——
    def test_score(self, desc: str) -> float:
        fam, model = self._lookup(desc)
        return self._score(self._code(fam, model), "test")

    def _lookup(self, desc):
        for d, fam, model in OPERATOR_SETS[self.opset]:
            if d == desc:
                return fam, model
        raise KeyError(desc)


# ---------- 2. 两个被试：L1（自治）vs L2（闸门接上） ----------

def agent_L1(env: MLCodeEnv):
    """L1：人类"在环"但闸没接上 = 实际上离环。选验证集冠军，直接 ship。"""
    candidates = [d for d, _, _ in OPERATOR_SETS[env.opset]]
    scored = {d: env.val_score(d) for d in candidates}  # ← 只看了 val
    best = max(scored, key=scored.get)
    return best, scored[best]   # (ship 的模型, 它的验证分)


class HITLGate:
    """卡在 L1→L2 之间的可信闸门。

    决策规则（可解释、可复现）：
    1. 拿 agent 提议的模型，在测试集上打分；
    2. 重审整个候选池的测试分，找出"测试冠军"；
    3. **主信号——冠军易主**：若 agent 提议的 ≠ 测试冠军（验证集说了谎）→ 拦截；
    4. **次信号——严重落差**：验证→测试落差超过 overfit_threshold（默认 0.05，
       即 5 个百分点），提示可能过拟合，也拦截。
       两条都不触发 → 放行，ship agent 提议的模型。
    注意：普通的 1~2pp 泛化落差（小测试集的正常现象）不算"说谎"，不触发——这正是
    闸门"校准良好、不捣乱"的体现。
    """

    def __init__(self, env: MLCodeEnv, overfit_threshold: float = 0.05):
        self.env = env
        self.overfit_threshold = overfit_threshold

    def audit(self, proposed: str):
        candidates = [d for d, _, _ in OPERATOR_SETS[self.env.opset]]
        # agent 提议模型的测试分
        proposed_test = self.env.test_score(proposed)
        # 闸门重审整池 → 测试冠军（人类复核时会重新比较所有候选）
        test_scores = {d: self.env.test_score(d) for d in candidates}
        test_best = max(test_scores, key=test_scores.get)
        gap = self.env.val_score(proposed) - proposed_test  # 验证→测试落差

        flipped = (proposed != test_best) or (gap > self.overfit_threshold)
        if flipped:
            decision = "BLOCK"          # 拦截，交还人类
            ship = test_best            # 人类复核后应选测试冠军
            ship_test = test_scores[test_best]
        else:
            decision = "PASS"           # 放行，验证集可信
            ship = proposed
            ship_test = proposed_test
        return {
            "decision": decision,
            "proposed": proposed,
            "proposed_test": proposed_test,
            "gap": gap,
            "test_best": test_best,
            "ship": ship,
            "ship_test": ship_test,
        }


# ---------- 3. 跑实验 ----------

def run_scenario(opset: str):
    env = MLCodeEnv(opset, seed=1)
    proposed, proposed_val = agent_L1(env)
    gate = HITLGate(env)
    report = gate.audit(proposed)

    print(f"  ── 算子集合 = {opset} "
          f"({'能标准化' if opset == 'strong' else '只能调超参'}) ──")
    print(f"    L1(自治):   ship {proposed:<8s}  验证={proposed_val:.4f} "
          f"→ 真实测试={env.test_score(proposed):.4f}")
    print(f"    闸门:       {report['decision']}  "
          f"(验证→测试落差 {report['gap']:+.4f}, "
          f"测试冠军={report['test_best']})")
    if report["decision"] == "BLOCK":
        print(f"               → 交还人类：改 ship {report['ship']} "
              f"(真实测试 {report['ship_test']:.4f})")
    else:
        print(f"               → 放行：验证集可信，ship {report['ship']} "
              f"(真实测试 {report['ship_test']:.4f})")
    print()
    return proposed, report


if __name__ == "__main__":
    print("HITL 闸门实验：同环境，比较 L1(自治) 与 L2(闸门接上)\n")
    _, weak = run_scenario("weak")
    _, strong = run_scenario("strong")

    print("  结论（全部实测）：")
    print(f"    · weak  ：L1 蒙头 ship 了验证冠军 {weak['proposed']}，"
          f"真实测试只有 {weak['proposed_test']:.4f}；")
    print(f"            闸门拦截，指出测试冠军其实是 {weak['test_best']} "
          f"({weak['ship_test']:.4f})——拦下了一个次优决策。")
    print(f"    · strong：L1 与 L2 选了同一模型 {strong['ship']} "
          f"(真实测试 {strong['ship_test']:.4f})，闸门放行——校准良好，不捣乱。")
    print("    · 一句话：闸门的价值不在'更准'，而在'验证集说谎时拦住你'。")
    print("    · 这正楔进 MLGym 的 L1→L2：把一道独立审计信号，变成 agent 升级自主性的门票。")
