"""SFT 数据集：读 jsonl，套模板，构造 batch。

数据格式（每行一个 json）：
    {"instruction": "Who are you?", "response": "I am ChatGPT."}
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from template import build_example  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# GPT-2 的 <|endoftext|>，既当终止符也当 padding
PAD_ID = 50256


def load_jsonl(path: Path) -> list[dict]:
    """读 jsonl 数据集。"""
    examples = []
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{lineno} 不是合法 json: {exc}") from exc

            for field in ("instruction", "response"):
                if field not in obj:
                    raise ValueError(f"{path}:{lineno} 缺少字段 {field!r}")

            examples.append(
                {"instruction": obj["instruction"], "response": obj["response"]}
            )

    if not examples:
        raise ValueError(f"{path} 里没有数据")

    return examples


def encode_dataset(examples: list[dict], enc, max_len: int) -> tuple[list, list]:
    """把全部样本套模板编码，超长的丢掉。

    返回 (inputs, labels)，各自是等长的 list[list[int]]。
    """
    encoded = []
    dropped = 0

    for ex in examples:
        ids, labels = build_example(enc, ex["instruction"], ex["response"])
        if len(ids) > max_len:
            dropped += 1
            continue
        encoded.append((ids, labels))

    if dropped:
        print(f"  [警告] {dropped} 条样本超过 max_len={max_len}，已丢弃")

    if not encoded:
        raise ValueError("所有样本都被丢弃了，检查 max_len 设置")

    return [e[0] for e in encoded], [e[1] for e in encoded]


def make_batch(
    inputs: list[list[int]],
    labels: list[list[int]],
    indices: list[int],
    pad_id: int = PAD_ID,
    device: str = "cpu",
    rng: random.Random | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """按 indices 取一个 batch，右 padding 到本 batch 内最长。

    返回 (x, y)，**两者已经错开一位**：
        x = ids[:-1]
        y = labels[1:]

    ⚠️ 为什么必须错位
    ----------------
    GPT 是自回归的：位置 i 的输出 logits[i] 预测的是**第 i+1 个** token。
    而 model.py 的 loss 是逐位置对齐比较的：

        cross_entropy(logits[i], y[i])

    所以 y[i] 必须放「第 i+1 个 token」，也就是 labels[i+1]。

    如果直接把 labels 原样传进去，模型学到的就是「复制当前 token」——
    一个平凡任务，loss 会迅速归零，但生成的文本完全是垃圾。
    **训练指标完美、产出全错**，这是最难发现的一类 bug。

    ⚠️ padding 的位置填 -100，不能填 pad_id
    ----------------------------------------
    填 pad_id 等于在教模型「预测填充符」，会污染 loss。
    """
    rng = rng or random

    # 先错位，再 padding
    seqs = [(inputs[i][:-1], labels[i][1:]) for i in indices]
    seqs = [(x, y) for x, y in seqs if len(x) > 0]
    if not seqs:
        raise ValueError("batch 里所有样本长度都只有 1，无法错位，检查数据")

    max_len = max(len(x) for x, _ in seqs)

    padded_in, padded_lab = [], []
    for x, y in seqs:
        pad_n = max_len - len(x)
        padded_in.append(x + [pad_id] * pad_n)
        padded_lab.append(y + [-100] * pad_n)

    return (
        torch.tensor(padded_in, dtype=torch.long, device=device),
        torch.tensor(padded_lab, dtype=torch.long, device=device),
    )


def report(examples: list[dict], inputs: list[list[int]], labels: list[list[int]]) -> None:
    """打印数据集统计。"""
    lengths = [len(x) for x in inputs]
    train_tokens = sum(1 for labs in labels for x in labs if x != -100)
    total_tokens = sum(lengths)

    print(f"\n  样本数        {len(inputs)}")
    print(f"  总 token      {total_tokens:,}")
    print(f"  参与 loss     {train_tokens:,}  ({train_tokens / total_tokens:.1%})")
    print(f"  长度 min/max  {min(lengths)} / {max(lengths)}")
    print(f"  → 被 mask 掉的 {total_tokens - train_tokens:,} 个 token 是提问部分")


if __name__ == "__main__":
    import argparse
    import tiktoken

    from template import inspect

    parser = argparse.ArgumentParser(description="检查 SFT 数据集")
    parser.add_argument("path", type=Path, help="jsonl 文件")
    parser.add_argument("--max-len", type=int, default=512)
    parser.add_argument("--show", type=int, default=3, help="展开看几条样本的 mask")
    args = parser.parse_args()

    enc = tiktoken.get_encoding("gpt2")
    examples = load_jsonl(args.path)
    inputs, labels = encode_dataset(examples, enc, args.max_len)

    print(f"\n数据集: {args.path}")
    report(examples, inputs, labels)

    for ex in examples[: args.show]:
        inspect(enc, ex["instruction"], ex["response"])
