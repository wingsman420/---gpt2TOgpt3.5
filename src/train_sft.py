"""SFT 训练循环。

从官方 GPT-2 权重出发，用带 loss mask 的数据做指令微调。

冒烟测试（一条数据，验证管线是否跑通）：
    .venv/Scripts/python.exe src/train_sft.py ^
        --data data/raw/smoke.jsonl --steps 300 --lr 3e-4 --out checkpoints/smoke.pt

正式训练（阶段 4，数万条数据）：
    .venv/Scripts/python.exe src/train_sft.py ^
        --data data/processed/train.jsonl --epochs 3 --lr 5e-5 --out checkpoints/sft.pt
"""

from __future__ import annotations

import argparse
import math
import random
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from load_weights import DEFAULT_LOCAL_DIR, load_model  # noqa: E402
from sft_data import encode_dataset, load_jsonl, make_batch, report  # noqa: E402
from template import inspect  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def get_lr(step: int, total: int, warmup: int, max_lr: float, min_lr: float) -> float:
    """线性 warmup + cosine 衰减。

    ⚠️ 微调的学习率必须比预训练小 1~2 个数量级。
       GPT-2 预训练用的是 6e-4，微调用那个会灾难性遗忘 ——
       通用语言能力被冲掉，模型变成只会背你这几条数据的复读机。
    """
    if step < warmup:
        return max_lr * (step + 1) / max(warmup, 1)
    if step >= total:
        return min_lr
    ratio = (step - warmup) / max(total - warmup, 1)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * ratio))


def build_optimizer(model, lr: float, weight_decay: float):
    """参数分两组：权重（二维以上）加 weight decay，bias 和 LayerNorm 不加。"""
    decay, no_decay = [], []
    for _, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (decay if p.dim() >= 2 else no_decay).append(p)

    groups = [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    n_decay = sum(p.numel() for p in decay)
    n_nodecay = sum(p.numel() for p in no_decay)
    print(f"  优化器参数分组: 加衰减 {n_decay:,}  不加衰减 {n_nodecay:,}")

    return torch.optim.AdamW(groups, lr=lr, betas=(0.9, 0.95), eps=1e-8)


def main() -> None:
    parser = argparse.ArgumentParser(description="GPT-2 指令微调 (SFT)")
    parser.add_argument("--data", type=Path, required=True, help="训练数据 jsonl")
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "checkpoints" / "sft.pt")
    parser.add_argument("--local-dir", type=Path, default=DEFAULT_LOCAL_DIR, help="GPT-2 权重的本地目录")

    parser.add_argument("--steps", type=int, default=0, help="总步数（指定则覆盖 epochs）")
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--max-len", type=int, default=512)

    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--min-lr", type=float, default=0.0, help="cosine 衰减的下界")
    parser.add_argument("--warmup", type=int, default=0, help="warmup 步数，0 表示自动取 5%")
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--clip", type=float, default=1.0)

    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-amp", action="store_true", help="关闭 bf16 混合精度")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = (device == "cuda") and not args.no_amp

    print("=" * 74)
    print("  SFT 训练")
    print("=" * 74)
    print(f"\n  设备        {device}")
    print(f"  混合精度    {'bf16' if use_amp else '关闭'}")

    # ---------------- 数据 ----------------
    import tiktoken

    enc = tiktoken.get_encoding("gpt2")

    print(f"\n[1/4] 加载数据 {args.data}")
    examples = load_jsonl(args.data)
    inputs, labels = encode_dataset(examples, enc, args.max_len)
    report(examples, inputs, labels)

    # 训练前先人肉确认 mask 正确 —— 这一步永远不要跳
    inspect(enc, examples[0]["instruction"], examples[0]["response"])

    # ---------------- 模型 ----------------
    print(f"\n[2/4] 加载 GPT-2 基础权重")
    model = load_model(args.local_dir, device=device)
    model.train()  # ← 从 eval 切回 train（GPT-2 dropout=0，但语义上必须切）

    n_params = model.num_params()
    print(f"  可训练参数  {n_params:,}")

    # ---------------- 优化器 ----------------
    n = len(inputs)
    if args.steps > 0:
        total_steps = args.steps
    else:
        total_steps = max(1, int(math.ceil(n / args.batch_size * args.epochs)))
    warmup = args.warmup if args.warmup > 0 else max(1, int(total_steps * 0.05))

    print(f"\n[3/4] 配置优化器")
    optimizer = build_optimizer(model, args.lr, args.weight_decay)
    print(f"  总步数      {total_steps}  (warmup {warmup})")
    print(f"  lr          {args.lr} -> {args.min_lr}")
    print(f"  batch       {args.batch_size} x accum {args.grad_accum} = {args.batch_size * args.grad_accum}")

    # ---------------- 训练 ----------------
    print(f"\n[4/4] 开始训练\n")
    print(f"  {'step':>6}  {'loss':>8}  {'lr':>10}  {'耗时':>8}")
    print(f"  {'-' * 40}")

    t0 = time.time()
    losses: list[float] = []
    rng = random.Random(args.seed)

    for step in range(total_steps):
        lr = get_lr(step, total_steps, warmup, args.lr, args.min_lr)
        for group in optimizer.param_groups:
            group["lr"] = lr

        # 采样一个 batch。数据量小于 batch_size 时会有放回地取。
        indices = rng.choices(range(n), k=min(args.batch_size, n)) if n < args.batch_size \
            else rng.sample(range(n), args.batch_size)
        x, y = make_batch(inputs, labels, indices, device=device, rng=rng)

        with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=use_amp):
            _, loss = model(x, targets=y)

        (loss / args.grad_accum).backward()

        if (step + 1) % args.grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        losses.append(loss.item())

        if step % args.log_every == 0 or step == total_steps - 1:
            recent = sum(losses[-args.log_every :]) / len(losses[-args.log_every :])
            elapsed = time.time() - t0
            print(f"  {step:>6}  {recent:>8.4f}  {lr:>10.2e}  {elapsed:>7.1f}s")

    # ---------------- 保存 ----------------
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "config": model.config,
            "step": total_steps,
            "final_loss": sum(losses[-20:]) / min(20, len(losses)),
            "data": str(args.data),
            "lr": args.lr,
        },
        args.out,
    )

    first = sum(losses[:20]) / min(20, len(losses))
    last = sum(losses[-20:]) / min(20, len(losses))
    print(f"\n{'=' * 74}")
    print(f"  完成")
    print(f"{'=' * 74}")
    print(f"\n  loss      {first:.4f}  ->  {last:.4f}")
    print(f"  检查点    {args.out}")
    print(f"\n  验证：")
    print(f"    .venv/Scripts/python.exe src/chat.py --ckpt {args.out}")


if __name__ == "__main__":
    main()
