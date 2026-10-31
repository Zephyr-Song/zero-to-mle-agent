"""
L01 · ReAct 循环：所有 Agent 的骨架
====================================

跑法：python lessons/l01_react_loop.py   （纯标准库，无需任何依赖/API key）

【本课目标】
用不到 60 行代码，写出所有 LLM Agent 共用的那个骨架：ReAct 循环。
Think(思考) → Act(行动) → Observe(观察)，循环，直到任务完成。

【为什么从这开始】
你在这个仓库读过的论文，底层全是这个循环：
- AIDE 的方案树：每个节点就是一次"Act + Observe"（写代码 → 跑分）
- ML-Agent 的 step-wise RL：奖励就是在循环的每一步上打的
- 区别只在于：谁来做决策（真 LLM / 训练出的小模型），以及动作空间长什么样

【本课的关键设计决定】
MockLLM **没有任何内部记忆**——它每一步的决策只依赖传入的 context（对话历史）。
这是刻意为之：真 LLM 也是无状态的，所谓"记忆"就是把历史拼进 prompt 再发一次。
理解了这一点，你就理解了后面所有课要解决的"上下文爆炸"问题从哪来。

【思考题】（答案都在下一课）
1. 如果 MockLLM 改成随机猜，平均要几轮才能猜中？为什么二分只要 ≤7 轮？
2. 环境只返回"高了/低了"三个字——这种"信息极少的观察"对 agent 是好事还是坏事？
3. 把 max_steps 从 10 改成 3，循环退出时 agent 并没有完成任务。真实系统里
   （回想 MLE-Bench 的 24h 时限）这类"预算耗尽"该怎么处理？

【下一课预告】
L02：给 agent 换一个真正要"写代码"的环境——它会自己写 ML 代码、
去环境里跑分，第一次拿到真实的准确率。
"""

import random

# ---------- 1. 环境：agent 能"做"的事，以及世界的反馈 ----------

class NumberEnv:
    """藏一个 1~100 的整数，只回报"高了/低了/正确"。
    agent 看不到 secret——就像真实 ML 环境里，agent 看不到测试集标签。"""

    def __init__(self, secret: int):
        self.secret = secret
        self.done = False

    def step(self, action: str) -> str:
        n = int(action.split(":", 1)[1])
        if n == self.secret:
            self.done = True
            return f"{n} 正确！"
        return f"{n} 高了，往小猜" if n > self.secret else f"{n} 低了，往大猜"


# ---------- 2. MockLLM：用规则模拟"大脑"，接口与真 LLM 完全一致 ----------

class MockLLM:
    """输入：完整对话历史 context；输出：(思考, 动作)。
    刻意不维护任何内部状态——决策只看 context（和真 LLM 一样是无状态的）。"""

    def __call__(self, context: str):
        # 从历史里恢复"我还剩哪些候选"：逐行累积解析所有 (动作, 观察) 对。
        # ⚠️ 常见 Bug：只看最后一对——之前轮次的约束会全部丢失（范围越修越错）。
        lo, hi, pending = 1, 100, None
        for line in context.splitlines():
            if line.startswith("动作: GUESS: "):
                pending = int(line.rsplit(":", 1)[1])
            elif line.startswith("观察: ") and pending is not None:
                obs = line[len("观察: "):]
                if "正确" in obs:
                    return (f"{pending} 已确认正确", f"FINISH: {pending}")
                if "高" in obs:
                    hi = pending - 1
                elif "低" in obs:
                    lo = pending + 1
                pending = None  # 该对已消费
        mid = (lo + hi) // 2
        return (f"剩余范围 [{lo}, {hi}]，取中点 {mid}（二分查找）", f"GUESS: {mid}")


# ---------- 3. ReAct 循环：本课的主角 ----------

def react_loop(task: str, llm, env, max_steps: int = 10):
    """Think → Act → Observe，直到环境说 done 或预算耗尽。"""
    context = f"任务: {task}\n"
    print(context, end="")
    for i in range(1, max_steps + 1):
        thought, action = llm(context)                    # Think + Act（真 LLM 一次输出两样）
        context += f"[第{i}轮] 思考: {thought}\n动作: {action}\n"
        if action.startswith("FINISH"):                   # agent 主动宣布完成
            answer = action.split(":", 1)[1].strip()
            print(f"[第{i}轮] ✅ FINISH → 答案: {answer}")
            return answer
        obs = env.step(action)                            # Observe
        context += f"观察: {obs}\n"
        print(f"[第{i}轮] 思考: {thought}\n         动作: {action} → 观察: {obs}")
    print("预算耗尽，任务未完成（见思考题 3）")
    return None


if __name__ == "__main__":
    random.seed(7)  # 固定种子，保证每次运行可复现
    env = NumberEnv(secret=random.randint(1, 100))
    task = "我心里藏了一个 1~100 的整数。用 `GUESS: n` 来猜；确认猜中后用 `FINISH: n` 给出答案。"
    answer = react_loop(task, MockLLM(), env, max_steps=10)
    print(f"\n{'='*46}\n结果: {answer} · 总轮数见上方 [第N轮]")
