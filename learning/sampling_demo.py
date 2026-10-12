"""temperature 和 top_k 到底对 logits 做了什么。

用真实 GPT-2 的 logits，把采样的四步逐级展开，
让你看清每个参数在哪一步、改了什么数字。

用法
====
    .venv/Scripts/python.exe learning/sampling_demo.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

# 本文件在 learning/ 下，推理主入口在 ../src/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from gpt2 import load  # noqa: E402

PROMPT = "The capital of France is"


def banner(t: str) -> None:
    print("\n" + "=" * 74)
    print(f"  {t}")
    print("=" * 74)


def top_entries(logits: torch.Tensor, enc, n: int = 8):
    """返回 logits 里最大的 n 项，格式 [(文本, 原始分数, softmax 概率)]。"""
    probs = F.softmax(logits, dim=-1)
    v, i = torch.topk(logits, n)
    return [(enc.decode([t]), float(s), float(p)) for s, p, t in zip(v, probs[i], i)]


def show(label: str, logits: torch.Tensor, enc, n: int = 8, width: int = 40) -> None:
    """打印 top-n。柱状条按当前这一组里的最大概率归一化，方便横向对比形状。"""
    print(f"\n  {label}")
    entries = top_entries(logits, enc, n)
    top_p = max((p for _, _, p in entries), default=0.0)

    for text, score, prob in entries:
        bar = "█" * (int(prob / top_p * width) if top_p > 0 else 0)
        print(f"    {text!r:<14} logit {score:>9.2f}   p {prob:.4f}  {bar}")


def main() -> None:
    model, enc = load()
    device = next(model.parameters()).device

    ids = torch.tensor([enc.encode(PROMPT)], dtype=torch.long, device=device)
    with torch.no_grad():
        logits, _ = model(ids)

    raw = logits[0, -1].clone()  # [50257] 最后一个位置的打分

    # ==================================================================
    banner("第 0 步：模型吐出来的原始 logits")

    print(f"""
  prompt: {PROMPT!r}
  模型输出 logits 形状 {tuple(logits.shape)}，我们只看最后一个位置：
  {tuple(raw.shape)} 个数，对应词表里的 50257 个 token。

  注意这些是**未归一化的分数**，可正可负，加起来不是 1。
""")
    show("原始 logits 的 top 8：", raw, enc)

    print(f"""
  问题：怎么从这些分数里选一个 token？

  最笨的办法是直接 argmax（永远选最大的）—— 结果是死的，同一个 prompt
  永远给同一句话，而且容易陷入重复循环。

  要**随机采样**，就得先把分数变成概率分布。这一步是 softmax。
  而 temperature 就是插在 softmax 前面的一个旋钮。
""")

    # ==================================================================
    banner("第 1 步：temperature —— softmax 之前先除一下")

    print(f"""
  代码：  logits = logits[:, -1, :] / temperature

  softmax 的定义：

      p_i = exp(x_i / T) / Σ_j exp(x_j / T)

  除在 exp 里面，所以 T 直接改变分布的**尖锐程度**。
""")

    for temp in (0.3, 1.0, 1.5):
        show(f"temperature = {temp}   (代码: raw / {temp})", raw / temp, enc, n=5)

    print(f"""
  T < 1：分数差距被放大 → 大的更大 → 分布尖锐 → 保守、重复
  T = 1：原样
  T > 1：分数差距被压平 → 分布接近均匀 → 容易抽到低概率词 → 胡说

  极端情况：
    T → 0   分布变成一个尖峰，等价于 argmax（贪心解码）
    T → ∞  分布变成均匀分布，完全随机
""")

    # ==================================================================
    banner("第 2 步：top_k —— 把长尾直接砍掉")

    print(f"""
  代码：  v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
          logits[logits < v[:, [-1]]] = -float("inf")

  拆开看：
    torch.topk(logits, k)   返回前 k 大的**值和索引**
                            v 形状 [50257] -> topk 之后 [k]
    v[:, [-1]]              取第 k 大的那个值，作为**门槛**
    logits < 门槛           所有比它小的位置
    = -inf                  赋成负无穷
""")

    # 真实演示
    k = 40
    x = raw.clone()
    v, idx = torch.topk(x, k)
    threshold = v[-1]
    n_killed = (x < threshold).sum().item()

    print(f"  实测（k={k}）：")
    print(f"    第 1 大的值      {v[0]:.4f}   ({enc.decode([idx[0]])!r})")
    print(f"    第 {k} 大的值    {threshold:.4f}   ({enc.decode([idx[-1]])!r})  ← 门槛")
    print(f"    低于门槛的 token {n_killed:,} 个被置为 -inf")
    print(f"    剩下的候选       {50257 - n_killed:,} 个")

    print(f"\n  为什么 v[:, [-1]] 要写成 [[-1]] 而不是 [-1]？")
    print(f"    v 的形状是 [B, k]。v[:, -1]   -> [B]      广播时会和词的维度对齐，错")
    print(f"                       v[:, [-1]] -> [B, 1]   正确地按批次对齐")
    print(f"    多一个方括号 = 多保留一个维度，这是 PyTorch 广播的经典坑。")

    for kk in (1, 5, 50, 50257):
        x = raw.clone()
        v, _ = torch.topk(x, min(kk, x.numel()))
        x[x < v[-1]] = float("-inf")
        kept = (x > float("-inf")).sum().item()
        show(f"top_k = {kk}  (保留 {kept:,} 个候选)", x, enc, n=4)

    print(f"""
  top_k = 1 时只剩一个候选，softmax 后概率恒为 1 —— 又变回贪心解码。
  top_k = 50257 等于不截断。

  为什么需要它：temperature 只调整形状，**管不住长尾**。
  50257 个 token 里有一大堆垃圾 token，单个概率极低，
  但乘起来不可忽略，偶尔抽中一个整句话就崩了。top_k 直接把它们清零。
""")

    # ==================================================================
    banner("第 3 步：softmax —— -inf 变成概率 0")

    x = raw.clone() / 0.8
    v, _ = torch.topk(x, 40)
    x[x < v[-1]] = float("-inf")
    probs = F.softmax(x, dim=-1)

    print(f"""
  代码：  probs = F.softmax(logits, dim=-1)

  被置为 -inf 的位置，exp(-inf) = 0，softmax 后概率恰好是 0。
  **永远不会被抽中。** 这就是 top-k 生效的机制。
""")
    print(f"  检查：")
    print(f"    概率为 0 的 token  {int((probs == 0).sum()):,} 个")
    print(f"    概率之和           {probs.sum().item():.10f}")
    print(f"    最大概率           {probs.max().item():.4f}")

    show(f"softmax 之后 (T=0.8, k=40)", x, enc, n=6)

    # ==================================================================
    banner("第 4 步：multinomial —— 按概率抽签")

    print(f"""
  代码：  idx_next = torch.multinomial(probs, num_samples=1)

  这一步才是真正的「随机」。它按 probs 给定的概率抽一个下标：
  概率 0.3 的 token，长期看有 30% 的几率被抽中。

  注意：**逐 token 独立采样**。每生成一个 token 就重新抽一次签，
  所以同一个 prompt 跑两次结果不同（除非 temperature 极低）。
""")

    n_samples = 3000
    counts = torch.bincount(
        torch.multinomial(probs, num_samples=n_samples, replacement=True),
        minlength=probs.numel(),
    )
    print(f"  模拟抽签 {n_samples} 次，实际比例 vs 理论概率：")
    v, i = torch.topk(probs, 6)
    for prob, tok in zip(v.tolist(), i.tolist()):
        actual = counts[tok].item() / n_samples
        print(f"    {enc.decode([tok])!r:<14} 理论 {prob:.4f}   实际 {actual:.4f}")

    # ==================================================================
    banner("总结：四步流水线")

    print(f"""
      logits  [50257]   ← 模型输出，未归一化的分数
         │
         │  / temperature        ← 调「形状」：尖锐还是平坦
         ▼
      logits  [50257]
         │
         │  topk + 置 -inf        ← 调「范围」：砍掉长尾垃圾
         ▼
      logits  [50257]（其中大部分是 -inf）
         │
         │  softmax              ← -inf 变 0，其余归一化到和为 1
         ▼
      probs   [50257]  （和 = 1，大量位置是 0）
         │
         │  multinomial          ← 按概率抽一个
         ▼
      下一个 token

  temperature 和 top_k 分工不同：
    temperature  改变**保留下来的那些** token 之间的相对概率
    top_k        决定**哪些** token 有机会被保留

  ⚠️ 两者都只是**采样技巧**，不改变模型的判断。
     模型认为最可能的是 ' the'，你怎么调参数它都还是认为 ' the' 最可能。
     调参数只能改变「你多大概率听它的」，不能让模型变得更聪明。

     **这正是指令微调要解决的问题 —— 那是改模型，不是改采样。**
""")


if __name__ == "__main__":
    main()
