"""跟（微调过的）GPT-2 对话。

用法
====
    # 只跑原始 GPT-2，看它有多不会聊天
    .venv/Scripts/python.exe src/chat.py --base

    # 跑微调后的模型
    .venv/Scripts/python.exe src/chat.py --ckpt checkpoints/smoke.pt

    # 并排对比（冒烟测试的验收方式）
    .venv/Scripts/python.exe src/chat.py --ckpt checkpoints/smoke.pt --compare

    # 交互模式
    .venv/Scripts/python.exe src/chat.py --ckpt checkpoints/smoke.pt -i
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _sft.template import EOS_TEXT, build_prompt  # noqa: E402
from gpt2 import DEFAULT_LOCAL_DIR, load  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def respond(
    model,
    enc,
    instruction: str,
    max_new_tokens: int = 30,
    temperature: float = 0.8,
    top_k: int = 40,
) -> str:
    """给一句提问，返回模型的回复正文（不含模板）。"""
    prompt = build_prompt(instruction)
    device = next(model.parameters()).device
    ids = torch.tensor([enc.encode(prompt)], dtype=torch.long, device=device)

    model.eval()
    with torch.no_grad():
        out = model.generate(
            ids, max_new_tokens, temperature=temperature, top_k=top_k
        )

    full = enc.decode(out[0].tolist())

    # 只取 "### Response:" 之后的部分
    marker = "### Response:\n"
    text = full.split(marker, 1)[1] if marker in full else full

    # 遇到终止符就截断
    text = text.split(EOS_TEXT)[0]
    return text.strip()


def load_checkpoint(model, ckpt_path: Path) -> int:
    """把微调权重灌进模型，返回训练步数。"""
    # 检查点里只有基础类型和张量（见 train_sft.py 保存处），
    # 所以可以开 weights_only=True —— 它禁止 pickle 执行任意代码，更安全。
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
    missing = [k for k in missing if k != "lm_head.weight"]
    if missing or unexpected:
        print(f"[!] 检查点加载异常: 缺失 {missing} 多余 {unexpected}")

    n_step = ckpt.get("step", -1)
    loss = ckpt.get("final_loss", float("nan"))
    print(f"[OK] 已加载检查点 {ckpt_path.name}（训练 {n_step} 步，loss {loss:.4f}）")
    return n_step


def ask_one(model, enc, question: str, args) -> None:
    answer = respond(
        model, enc, question, args.tokens, args.temperature, args.top_k
    )
    print(f"\n  问: {question}")
    print(f"  答: {answer}")


def main() -> None:
    parser = argparse.ArgumentParser(description="和 GPT-2 对话")
    parser.add_argument("--ckpt", type=Path, help="微调后的检查点")
    parser.add_argument("--base", action="store_true", help="只跑原始 GPT-2")
    parser.add_argument("--compare", action="store_true", help="微调前后并排对比")
    parser.add_argument("--local-dir", type=Path, default=DEFAULT_LOCAL_DIR)
    parser.add_argument("--prompt", action="append", help="要问的问题，可重复指定")
    parser.add_argument("-i", "--interactive", action="store_true")
    parser.add_argument("--tokens", type=int, default=30)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    args = parser.parse_args()

    if not args.ckpt and not args.base:
        parser.error("要么用 --base 跑原始模型，要么用 --ckpt 指定检查点")

    questions = args.prompt or ["Who are you?"]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- 微调后的模型 ---
    if args.ckpt:
        print("加载微调模型 ...")
        model_ft, enc = load(args.local_dir, device=device)
        load_checkpoint(model_ft, args.ckpt)

        print("\n" + "=" * 66)
        print("  微调后的 GPT-2")
        print("=" * 66)
        for q in questions:
            ask_one(model_ft, enc, q, args)

    # --- 并排对比 ---
    if args.base or args.compare:
        if args.ckpt and args.compare:
            print("\n把权重重置回原始 GPT-2 ...")
        else:
            print("加载原始 GPT-2 ...")
        model_base, enc = load(args.local_dir, device=device)

        label = "原始 GPT-2（对照组）" if args.compare else "原始 GPT-2"
        print("\n" + "=" * 66)
        print(f"  {label}")
        print("=" * 66)
        for q in questions:
            ask_one(model_base, enc, q, args)

    # --- 交互 ---
    if args.interactive:
        print("\n" + "=" * 66)
        print("  交互模式（Ctrl+C 或输入空行退出）")
        print("=" * 66)
        try:
            while True:
                q = input("\n  问: ").strip()
                if not q:
                    break
                target = model_ft if args.ckpt else model_base
                print(f"  答: {respond(target, enc, q, args.tokens, args.temperature, args.top_k)}")
        except (KeyboardInterrupt, EOFError):
            print()


if __name__ == "__main__":
    main()
