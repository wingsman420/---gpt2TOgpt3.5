"""与 HuggingFace 的 GPT-2 对拍，确认手写实现正确。

这是阶段 1 的验收脚本，也是整个实验的地基。
没有这一步，后面遇到「loss 不降」或「输出乱码」时，
你无法判断是 SFT 写错了还是模型本身就是错的 —— 会白白浪费好几天。

这一脚本是**唯一**允许 import transformers 的地方，
它只是调试工具，不进入训练管线。

用法
====
    .venv/Scripts/python.exe src/verify_parity.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from load_weights import DEFAULT_LOCAL_DIR, convert_state_dict, download  # noqa: E402
from model import GPT, GPTConfig  # noqa: E402

# 判定阈值。
#   权重转换正确  -> 差异约 1e-5 量级
#   转置/bias 错了 -> 差异约 1e-1 量级
#   激活函数用错  -> 差异约 1e-2 量级
# 1e-3 能干净地把「对」和「错」分开。
TOL = 1e-3

PROMPTS = [
    "Hello, my name is",
    "The capital of France is",
    "def fibonacci(n):",
    "In 1969, Apollo 11",
]


def build_model(activation: str, local_dir: Path) -> GPT:
    """用指定激活函数构造手写模型，并灌入官方权重。"""
    from safetensors.torch import load_file

    hf_config = json.loads((local_dir / "config.json").read_text(encoding="utf-8"))
    config = GPTConfig(
        block_size=hf_config.get("n_ctx", 1024),
        vocab_size=hf_config.get("vocab_size", 50257),
        n_layer=hf_config.get("n_layer", 12),
        n_head=hf_config.get("n_head", 12),
        n_embd=hf_config.get("n_embd", 768),
        activation=activation,
        dropout=0.0,
    )

    model = GPT(config)
    sd = convert_state_dict(load_file(str(local_dir / "model.safetensors"), device="cpu"))
    missing, unexpected = model.load_state_dict(sd, strict=False)
    missing = [k for k in missing if k != "lm_head.weight"]

    if missing or unexpected:
        print(f"[!] activation={activation} 权重加载异常")
        print(f"    缺失: {missing}")
        print(f"    多余: {unexpected}")

    model.eval()
    return model


def our_full_logits(model: GPT, ids: torch.Tensor) -> torch.Tensor:
    """取出手写模型在**所有位置**的 logits，形状 [B, T, vocab]。

    ⚠️ 这里踩过一个坑，值得记住：

    GPT.forward() 在 targets=None 时只返回最后一个位置的 logits（形状
    [B, 1, vocab]），这是生成时的快路径；而 HF 返回全部位置 [B, T, vocab]。

    直接相减会被 PyTorch 广播：把「我们的最后一个位置」和「HF 的每一个
    位置」逐一对比，误差直接爆到 85 以上，看起来像是模型全错了，
    实际上两个模型都是对的。**形状不一致的减法不会报错，这是最危险的地方。**

    这里传一个占位 target 走训练分支，就能拿到完整序列的 logits。
    """
    logits, _ = model(ids, targets=ids)  # loss 是副产品，不用
    return logits


def compare(model_mine: GPT, model_hf, tokenizer) -> float:
    """在固定输入上比较两个模型所有位置的 logits，返回最大绝对误差。"""
    max_diff = 0.0

    for prompt in PROMPTS:
        ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long)

        with torch.no_grad():
            logits_mine = our_full_logits(model_mine, ids)
            logits_hf = model_hf(ids).logits

        assert logits_mine.shape == logits_hf.shape, (
            f"形状不一致: 手写 {tuple(logits_mine.shape)} vs HF {tuple(logits_hf.shape)}"
            " —— 形状不同时相减会静默广播，必须在这里拦住"
        )

        diff = (logits_mine - logits_hf).abs().max().item()
        max_diff = max(max_diff, diff)

    return max_diff


def top5(model, tokenizer, prompt: str, is_hf: bool) -> list[tuple[str, float]]:
    """返回模型最看好的 5 个下一个 token。"""
    ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long)
    with torch.no_grad():
        if is_hf:
            logits = model(ids).logits[0, -1]
        else:
            logits, _ = model(ids)
            logits = logits[0, -1]

    top = torch.topk(logits, 5)
    return [
        (tokenizer.decode([i]), v)
        for v, i in zip(top.values.tolist(), top.indices.tolist())
    ]


def print_top5_side_by_side(model_mine, model_hf, tokenizer) -> None:
    """并排打印两个模型的预测。

    这是最直观的验收方式：如果权重转换错了，两边的预测会完全不同；
    如果对了，两边几乎一模一样。比看误差数字更容易发现问题。
    """
    print("\n" + "=" * 66)
    print("并排预测对比（下一个 token 的 top 5）")
    print("=" * 66)

    for prompt in PROMPTS[:2]:
        mine = top5(model_mine, tokenizer, prompt, is_hf=False)
        hf = top5(model_hf, tokenizer, prompt, is_hf=True)

        same = [t for t, _ in mine[:3]] == [t for t, _ in hf[:3]]
        print(f"\n  prompt: {prompt!r}   {'[一致]' if same else '[不一致!]'}")
        print(f"    手写: {' | '.join(f'{t!r} {v:6.2f}' for t, v in mine)}")
        print(f"    HF  : {' | '.join(f'{t!r} {v:6.2f}' for t, v in hf)}")


def main() -> None:
    local_dir = DEFAULT_LOCAL_DIR

    if not (local_dir / "model.safetensors").exists():
        download(local_dir=local_dir)

    try:
        import tiktoken
        from transformers import GPT2LMHeadModel
    except ImportError as exc:
        print(f"[FAIL] 缺少依赖: {exc}\n")
        print("请先运行:")
        print("    .venv/Scripts/python.exe -m pip install transformers")
        sys.exit(1)

    tokenizer = tiktoken.get_encoding("gpt2")

    # --- 用 CPU + float32 对拍，避免 GPU 浮点差异干扰判断 ---
    print("加载 HuggingFace 参考模型 ...")
    model_hf = GPT2LMHeadModel.from_pretrained(str(local_dir)).eval()

    # --- 逐个测试激活函数候选 ---
    # HF 的 gpt2 配置写的是 gelu_new（tanh 近似），但 GPT-2 论文用的是精确
    # GELU。哪个才对，实测说话，不靠记忆。
    print("\n" + "=" * 66)
    print("激活函数候选测试")
    print("=" * 66)

    results: dict[str, float] = {}
    models: dict[str, GPT] = {}

    for act in ("gelu_new", "gelu"):
        model = build_model(act, local_dir)
        models[act] = model

        diff = compare(model, model_hf, tokenizer)
        results[act] = diff
        mark = "[OK]  " if diff < TOL else "[不同]"
        print(f"  {mark} activation={act:<9} 最大误差 = {diff:.3e}")

    best_act = min(results, key=results.get)  # type: ignore[arg-type]
    best_diff = results[best_act]

    print("\n" + "=" * 66)
    print("结论")
    print("=" * 66)

    if best_diff >= TOL:
        print(f"\n[FAIL] 最好的激活函数 ({best_act}) 误差仍为 {best_diff:.3e}，超过阈值 {TOL}")
        print("\n说明还有别的地方不对。按以下顺序排查：")
        print("  1. 先确认两边 logits 形状一致（不一致会静默广播，误差虚高）")
        print("  2. c_proj 是方阵，转置与否形状都合法 —— 最容易漏的就是它")
        print("  3. 某个 bias 漏加载了（PyTorch 会静默跳过不匹配的 key）")
        print("  4. 权重绑定：lm_head.weight 必须与 wte.weight 是同一份")
        sys.exit(1)

    print(f"\n[PASS] 应使用 activation = {best_act!r}（误差 {best_diff:.3e}）")
    if best_act != models[best_act].config.activation:
        print(f"       注意：配置默认值是 {models[best_act].config.activation!r}，需要改")

    print_top5_side_by_side(models[best_act], model_hf, tokenizer)

    print("\n阶段 1 验收通过，可以进入阶段 2：证明「它不会聊天」。")


if __name__ == "__main__":
    main()
