"""SFT 训练循环。

从官方 GPT-2 权重出发，用带 loss mask 的数据做指令微调。

**日常用法：改 `src/_sft/config.py`，然后双击项目根目录的 `train.bat`。**

所有超参的默认值都来自配置文件，命令行只是可选的覆盖手段：

    :: 临时探路 200 步，其它参数照配置
    .venv/Scripts/python.exe src/train_sft.py --steps 200

冒烟测试（一条数据，验证管线是否跑通）：
    .venv/Scripts/python.exe src/train_sft.py ^
        --data data/raw/smoke.jsonl --steps 300 --lr 3e-4 --out checkpoints/smoke.pt
"""

from __future__ import annotations

import argparse
import math
import random
import sys
import time
from dataclasses import asdict
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _sft import config  # noqa: E402
from _sft.data import encode_dataset, load_jsonl, make_batch, report  # noqa: E402
from _sft.template import inspect  # noqa: E402
from gpt2 import load  # noqa: E402

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
    parser = argparse.ArgumentParser(
        description="GPT-2 指令微调 (SFT)",
        # 把每个参数的默认值（也就是 config.py 里的值）显示在 --help 里
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # 所有默认值都来自 src/_sft/config.py。
    # 命令行参数仍然可用，传了就覆盖配置文件 —— 但日常训练不用敲。
    parser.add_argument("--data", type=Path, default=config.DATA, help="训练数据 jsonl")
    parser.add_argument("--out", type=Path, default=config.OUT, help="检查点输出路径")
    parser.add_argument("--local-dir", type=Path, default=config.LOCAL_DIR, help="GPT-2 权重的本地目录")

    parser.add_argument("--steps", type=int, default=config.STEPS, help="总步数（>0 则覆盖 epochs；探路时用）")
    parser.add_argument("--epochs", type=float, default=config.EPOCHS, help="训练轮数")
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE, help="每步样本数")
    parser.add_argument("--grad-accum", type=int, default=config.GRAD_ACCUM, help="梯度累积步数，有效 batch = batch-size × 它")
    parser.add_argument("--max-len", type=int, default=config.MAX_LEN, help="单条样本最大 token 数，超长的丢弃")

    parser.add_argument("--lr", type=float, default=config.LR, help="学习率（微调必须比预训练小 1-2 个数量级）")
    parser.add_argument("--min-lr", type=float, default=config.MIN_LR, help="cosine 衰减的下界")
    # 注意：argparse 的 help 字符串走 % 格式化，字面量百分号必须写成 %%
    parser.add_argument("--warmup", type=int, default=config.WARMUP, help="warmup 步数，0 表示自动取 5%%")
    parser.add_argument("--weight-decay", type=float, default=config.WEIGHT_DECAY, help="权重衰减（只作用于权重矩阵）")
    parser.add_argument("--clip", type=float, default=config.CLIP, help="梯度裁剪的最大范数")

    parser.add_argument("--log-every", type=int, default=config.LOG_EVERY, help="每多少步打印一次日志")
    parser.add_argument("--seed", type=int, default=config.SEED, help="随机种子")
    parser.add_argument("--no-amp", action="store_true", default=config.NO_AMP, help="关闭 bf16 混合精度")
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
    model, _ = load(args.local_dir, device=device)
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
            # ⚠️ 存**普通 dict**，不要存 GPTConfig 对象本身。
            #
            # pickle 会把类的**模块路径**一起写进文件（原来是 "model.GPTConfig"）。
            # 一旦源文件改名或移动（比如 model.py -> _core/model.py），
            # 所有历史检查点就再也加载不了了 —— ModuleNotFoundError。
            #
            # 存成 asdict() 之后，检查点里只有基础类型 + 张量，两个好处：
            #   1. 源文件怎么重构都不影响老检查点
            #   2. torch.load 可以开 weights_only=True（更安全，见 chat.py）
            "config": asdict(model.config),
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
