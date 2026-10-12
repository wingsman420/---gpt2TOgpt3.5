"""把 Alpaca 数据集转成我们的 SFT 格式。

数据源：`tatsu-lab/alpaca`，52,002 条（HF 上存成 parquet）

原始字段
========
    instruction   任务描述
    input         任务素材，约 40% 的样本非空
    output        回答
    text          Alpaca 自己拼好的模板字符串 —— **我们不用**

为什么不用 `text`：它用的是 Alpaca 的模板（带一段 "Below is an instruction
that describes a task..." 前言），和我们 `template.py` 里的不一样。
**训练模板和推理模板必须逐字节一致**，所以模板一律由 `template.py` 生成。

关键决策：`input` 必须合并进 instruction
=========================================
丢掉 input 的话，指令就不完整了。例如：

    instruction = "Classify the following sentence as positive or negative."
    input       = "The movie was absolutely wonderful."
    output      = "positive"

不合并 → 模型看到"把下面这句话分类"却没有"下面这句话"，
只能学到"看到 classify 就输出 positive"。
合并后 → 模板变成：

    ### Instruction:
    Classify the following sentence as positive or negative.

    The movie was absolutely wonderful.

    ### Response:
    positive

用法
====
    .venv/Scripts/python.exe src/_sft/prepare_alpaca.py

产出
====
    data/processed/train.jsonl   约 51.5k 条
    data/processed/val.jsonl     500 条
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

# 本文件在 src/_sft/ 下，把 src/ 放进 sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from _sft.data import encode_dataset, load_jsonl, report  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

REPO_ID = "tatsu-lab/alpaca"
PARQUET_PATH = "data/train-00000-of-00001-a09b74b3ef9c3b56.parquet"

VAL_SIZE = 500
SEED = 42


def download(raw_dir: Path = RAW_DIR) -> Path:
    """下载 Alpaca 的 parquet 文件。

    HF 上的路径是 `data/train-xxx.parquet`，直接下会得到
    `data/raw/data/train-xxx.parquet` 这种嵌套结构，
    所以下完挪到 `data/raw/alpaca.parquet`。
    """
    os.environ.setdefault("HTTP_PROXY", "http://127.0.0.1:7897")
    os.environ.setdefault("HTTPS_PROXY", "http://127.0.0.1:7897")

    from huggingface_hub import hf_hub_download

    target = raw_dir / "alpaca.parquet"
    if target.exists():
        print(f"  已存在，跳过下载: {target}")
        return target

    raw_dir.mkdir(parents=True, exist_ok=True)
    print(f"  正在下载 {REPO_ID} ...")
    tmp = hf_hub_download(REPO_ID, PARQUET_PATH, repo_type="dataset", local_dir=str(raw_dir))

    # 挪到干净的位置
    target.write_bytes(Path(tmp).read_bytes())
    print(f"  下载完成: {target}  ({target.stat().st_size / 1024**2:.1f} MB)")
    return target


def merge_input(instruction: str, input_text: str) -> str:
    """把 input 合并进 instruction（空行分隔）。

    见模块文档里「关键决策」一节。
    """
    instruction = (instruction or "").strip()
    input_text = (input_text or "").strip()
    if not input_text:
        return instruction
    return f"{instruction}\n\n{input_text}"


def convert(rows: list[dict]) -> tuple[list[dict], dict[str, int]]:
    """转成 [{"instruction":..., "response":...}] 并统计丢弃情况。"""
    out: list[dict] = []
    dropped = {"output 为空": 0, "instruction 为空": 0}

    for row in rows:
        instruction = (row.get("instruction") or "").strip()
        response = (row.get("output") or "").strip()

        if not instruction:
            dropped["instruction 为空"] += 1
            continue
        if not response:
            dropped["output 为空"] += 1
            continue

        out.append(
            {
                "instruction": merge_input(instruction, row.get("input", "")),
                "response": response,
            }
        )

    return out, dropped


def write_jsonl(path: Path, examples: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")


def length_report(enc, examples: list[dict], sample: int = 5000) -> None:
    """打印 token 长度分布，用来决定 max_len 该设多少。"""
    import random as _random

    rows = _random.Random(0).sample(examples, min(sample, len(examples)))
    lengths = []
    for ex in rows:
        # 只算模板 + 提问 + 回复的完整长度
        text = f"### Instruction:\n{ex['instruction']}\n\n### Response:\n{ex['response']}"
        lengths.append(len(enc.encode(text)))

    lengths.sort()
    n = len(lengths)

    def pct(p: float) -> int:
        return lengths[min(n - 1, int(n * p))]

    print(f"\n  token 长度分布（抽样 {n:,} 条）")
    print(f"    min {lengths[0]:>5}    p25 {pct(0.25):>5}    中位 {pct(0.5):>5}")
    print(f"    p75 {pct(0.75):>5}    p90 {pct(0.90):>5}    p99 {pct(0.99):>5}    max {lengths[-1]:>7}")
    for ml in (256, 512, 1024):
        over = sum(1 for x in lengths if x > ml)
        print(f"    max_len={ml:<5} 会截断/丢弃 {over:>5} 条  ({over / n:.1%})")


def main() -> None:
    parser = argparse.ArgumentParser(description="Alpaca -> SFT jsonl")
    parser.add_argument("--val-size", type=int, default=VAL_SIZE)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--max-len", type=int, default=512,
                        help="编码时用来统计会被丢弃多少条")
    args = parser.parse_args()

    print("=" * 74)
    print("  Alpaca -> SFT 格式")
    print("=" * 74)

    # ---------------- 1. 下载 ----------------
    print(f"\n[1/5] 下载")
    parquet_path = download()

    # ---------------- 2. 读取 ----------------
    print(f"\n[2/5] 读取 parquet")
    import pyarrow.parquet as pq

    rows = pq.read_table(str(parquet_path)).to_pylist()
    print(f"  原始样本  {len(rows):,}")

    # ---------------- 3. 转换 ----------------
    print(f"\n[3/5] 转换")
    examples, dropped = convert(rows)
    print(f"  转换后    {len(examples):,}")
    for reason, n in dropped.items():
        if n:
            print(f"  丢弃      {n:,} 条（{reason}）")

    n_with_input = sum(1 for ex in examples if "\n\n" in ex["instruction"])
    print(f"  其中 {n_with_input:,} 条合并了 input 字段 ({n_with_input / len(examples):.1%})")

    # 抽一条合并了 input 的看看
    for ex in examples:
        if "\n\n" in ex["instruction"]:
            print(f"\n  ── 抽查一条有 input 的 ──")
            print(f"  instruction: {ex['instruction'][:150]!r}")
            print(f"  response   : {ex['response'][:80]!r}")
            break

    # ---------------- 4. 划分 ----------------
    print(f"\n[4/5] 划分 train / val")
    rng = random.Random(args.seed)
    rng.shuffle(examples)

    val = examples[: args.val_size]
    train = examples[args.val_size :]
    print(f"  train  {len(train):,}")
    print(f"  val    {len(val):,}  (seed={args.seed}，可复现)")

    train_path = PROCESSED_DIR / "train.jsonl"
    val_path = PROCESSED_DIR / "val.jsonl"
    write_jsonl(train_path, train)
    write_jsonl(val_path, val)
    print(f"\n  写入 {train_path}")
    print(f"  写入 {val_path}")

    # ---------------- 5. 统计 ----------------
    print(f"\n[5/5] 统计")
    import tiktoken

    enc = tiktoken.get_encoding("gpt2")
    length_report(enc, examples)

    inputs, labels = encode_dataset(train, enc, args.max_len)
    report(train, inputs, labels)

    print(f"\n{'=' * 74}")
    print("  完成")
    print(f"{'=' * 74}")
    print(f"\n  下一步：人肉检查 mask")
    print(f"    .venv/Scripts/python.exe src/_sft/data.py {train_path} --show 3")


if __name__ == "__main__":
    main()
