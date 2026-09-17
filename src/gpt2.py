"""原始 GPT-2：加载官方权重并生成文本。

**阶段 1 完成的标志。**

这是最干净的一份调用代码 —— 没有教学打印、没有对拍逻辑，
只有「加载模型」和「让它说话」两件事。

用法
====
    .venv/Scripts/python.exe src/gpt2.py "The capital of France is"
    .venv/Scripts/python.exe src/gpt2.py "Hello" --tokens 60

    # 也当模块用
    from gpt2 import load, complete
    model, enc = load()
    print(complete(model, enc, "Once upon a time"))
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from load_weights import DEFAULT_LOCAL_DIR, load_model  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load(local_dir: Path = DEFAULT_LOCAL_DIR, device: str = DEVICE):
    """加载 GPT-2 和分词器。返回 (model, encoder)。"""
    import tiktoken

    return load_model(local_dir, device=device), tiktoken.get_encoding("gpt2")


def complete(
    model,
    enc,
    prompt: str,
    max_new_tokens: int = 40,
    temperature: float = 0.8,
    top_k: int = 40,
) -> str:
    """给一段提示，返回 GPT-2 续写的完整文本（含原提示）。"""
    ids = torch.tensor([enc.encode(prompt)], dtype=torch.long, device=next(model.parameters()).device)
    out = model.generate(ids, max_new_tokens, temperature=temperature, top_k=top_k)
    return enc.decode(out[0].tolist())


def main() -> None:
    parser = argparse.ArgumentParser(description="用原始 GPT-2 续写文本")
    parser.add_argument("prompt", nargs="?", default="Hello, my name is")
    parser.add_argument("--tokens", type=int, default=40, help="生成多少个新 token")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    args = parser.parse_args()

    model, enc = load()

    print(f"\n{'─' * 60}")
    print(f"DEVICE      : {DEVICE}")
    print(complete(model, enc, args.prompt, args.tokens, args.temperature, args.top_k))
    print(f"{'─' * 60}\n")


if __name__ == "__main__":
    main()
