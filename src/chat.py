"""跟（微调过的）GPT-2 对话。

用法
====
    # 四条件对比（推荐）—— 同时看「模板的作用」和「微调的作用」
    .venv/Scripts/python.exe src/chat.py --ckpt checkpoints/sft.pt --compare

    # 只跑原始 GPT-2
    .venv/Scripts/python.exe src/chat.py --base

    # 只跑微调后的模型
    .venv/Scripts/python.exe src/chat.py --ckpt checkpoints/sft.pt

    # 交互模式
    .venv/Scripts/python.exe src/chat.py --ckpt checkpoints/sft.pt -i

关于 --compare 的四种组合
=========================
    ┌──────────────┬──────────────┬────────────────────────────┐
    │              │ 原始 prompt  │ 模板                       │
    ├──────────────┼──────────────┼────────────────────────────┤
    │ 原始权重     │     ①       │     ②                     │
    │ 微调权重     │     ③       │     ④                     │
    └──────────────┴──────────────┴────────────────────────────┘

    ①→②  模板的影响（同一个模型，换输入格式）
    ①→③  微调的影响（同一种输入，换权重）
    ②→④  微调的影响（同一种输入，换权重）

    为什么要全展示：base 模型没见过这个模板，套模板会让它退化
    （实测会陷入 "# Color" 复读）。只看 ② vs ④ 的话，你会分不清
    「微调让模型变好了」还是「base 被模板搞坏了」。四格一起看，
    两个因素就分离了。
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


def _generate(
    model,
    enc,
    text: str,
    max_new_tokens: int = 30,
    temperature: float = 0.8,
    top_k: int = 40,
) -> str:
    """给一段原始文本，返回**新生成的部分**（不含输入本身）。"""
    device = next(model.parameters()).device
    ids = torch.tensor([enc.encode(text)], dtype=torch.long, device=device)
    n_input = ids.shape[1]

    model.eval()
    with torch.no_grad():
        out = model.generate(
            ids, max_new_tokens, temperature=temperature, top_k=top_k
        )

    # 按 token 切，不按字符切 —— 避免解码/切片边界问题
    new_ids = out[0, n_input:].tolist()
    return enc.decode(new_ids)


def _clean(text: str) -> str:
    """截断到第一个 EOS。

    ⚠️ 模型有可能**第一个 token 就生成 EOS**（实测 "Who are you?" 用原始
    prompt 时约 1/5 的概率）。这时截断后是空字符串，如果直接打印，
    看起来像是程序坏了。所以给一个明确的提示。
    """
    body = text.split(EOS_TEXT)[0].strip()
    if not body:
        return "（空：模型第一个 token 就输出了 <|endoftext|>，没有产生内容）"
    return body


def respond(
    model,
    enc,
    instruction: str,
    max_new_tokens: int = 30,
    temperature: float = 0.8,
    top_k: int = 40,
) -> str:
    """套模板生成，返回回复正文（遇到 EOS 截断）。"""
    text = _generate(
        model, enc, build_prompt(instruction), max_new_tokens, temperature, top_k
    )
    return _clean(text)


def respond_raw(
    model,
    enc,
    instruction: str,
    max_new_tokens: int = 30,
    temperature: float = 0.8,
    top_k: int = 40,
) -> str:
    """**不套模板**，把 instruction 当原始文本直接喂进去。"""
    text = _generate(model, enc, instruction, max_new_tokens, temperature, top_k)
    return _clean(text)


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


def _print_answer(text: str, indent: str = "     ") -> None:
    for line in text.splitlines() or [""]:
        print(f"{indent}{line}")


def ask_one(model, enc, question: str, args) -> None:
    answer = respond(model, enc, question, args.tokens, args.temperature, args.top_k)
    print(f"\n  问: {question}")
    print(f"  答: {answer}")


def compare_four(model_ft, model_base, enc, questions, args) -> None:
    """四条件对比：分离「模板的作用」和「微调的作用」。"""
    print("\n" + "=" * 70)
    print("  四条件对比")
    print("=" * 70)
    print("   ① 原始权重 + 原始 prompt      ② 原始权重 + 模板")
    print("   ③ 微调权重 + 原始 prompt      ④ 微调权重 + 模板")
    print()
    print("   ①→②  模板的影响")
    print("   ①→③  微调的影响（同原始 prompt）")
    print("   ②→④  微调的影响（同模板）")

    for q in questions:
        print("\n" + "-" * 70)
        print(f"  问: {q}")
        print("-" * 70)

        for tag, model, use_template in (
            ("① 原始权重 + 原始 prompt", model_base, False),
            ("② 原始权重 + 模板", model_base, True),
            ("③ 微调权重 + 原始 prompt", model_ft, False),
            ("④ 微调权重 + 模板", model_ft, True),
        ):
            fn = respond if use_template else respond_raw
            answer = fn(model, enc, q, args.tokens, args.temperature, args.top_k)
            print(f"\n  {tag}")
            _print_answer(answer)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="和 GPT-2 对话",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--ckpt", type=Path, help="微调后的检查点")
    parser.add_argument("--base", action="store_true", help="只跑原始 GPT-2")
    parser.add_argument("--compare", action="store_true", help="四条件对比（需要 --ckpt）")
    parser.add_argument("--local-dir", type=Path, default=DEFAULT_LOCAL_DIR)
    parser.add_argument("--prompt", action="append", help="要问的问题，可重复指定")
    parser.add_argument("-i", "--interactive", action="store_true")
    parser.add_argument("--tokens", type=int, default=30)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    args = parser.parse_args()

    if not args.ckpt and not args.base:
        parser.error("要么用 --base 跑原始模型，要么用 --ckpt 指定检查点")
    if args.compare and not args.ckpt:
        parser.error("--compare 需要 --ckpt（没有检查点就没有可对比的微调模型）")

    questions = args.prompt or ["Who are you?"]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model_ft = model_base = None

    # ---------- 四条件对比 ----------
    if args.compare:
        print("加载微调模型 ...")
        model_ft, enc = load(args.local_dir, device=device)
        load_checkpoint(model_ft, args.ckpt)

        print("\n加载全新的原始 GPT-2（未经过任何微调）...")
        model_base, _ = load(args.local_dir, device=device)

        compare_four(model_ft, model_base, enc, questions, args)

    else:
        # ---------- 只跑微调模型 ----------
        if args.ckpt:
            print("加载微调模型 ...")
            model_ft, enc = load(args.local_dir, device=device)
            load_checkpoint(model_ft, args.ckpt)

            print("\n" + "=" * 66)
            print("  微调后的 GPT-2")
            print("=" * 66)
            for q in questions:
                ask_one(model_ft, enc, q, args)

        # ---------- 只跑原始模型 ----------
        if args.base:
            print("加载原始 GPT-2 ...")
            model_base, enc = load(args.local_dir, device=device)

            print("\n" + "=" * 66)
            print("  原始 GPT-2")
            print("=" * 66)
            for q in questions:
                ask_one(model_base, enc, q, args)

    # ---------- 交互 ----------
    if args.interactive:
        target = model_ft if model_ft is not None else model_base
        print("\n" + "=" * 66)
        print("  交互模式（Ctrl+C 或输入空行退出）")
        print("=" * 66)
        try:
            while True:
                q = input("\n  问: ").strip()
                if not q:
                    break
                print(f"  答: {respond(target, enc, q, args.tokens, args.temperature, args.top_k)}")
        except (KeyboardInterrupt, EOFError):
            print()


if __name__ == "__main__":
    main()
