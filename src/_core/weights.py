"""下载 HuggingFace 的 GPT-2 权重，转换后加载进手写模型。

唯一需要的变换：Conv1D -> Linear 的转置
=======================================

HF 的 GPT-2 用 Conv1D 实现线性层，它的权重形状是
    [in_features, out_features]
而 nn.Linear 是
    [out_features, in_features]
所以**必须转置**。

受影响的四个权重：

    attn.c_attn.weight   [768, 2304] -> [2304,  768]
    attn.c_proj.weight   [768,  768] -> [ 768,  768]   <- 方阵！
    mlp.c_fc.weight      [768, 3072] -> [3072,  768]
    mlp.c_proj.weight    [3072,  768] -> [ 768, 3072]

注意 c_proj 是**方阵**，转不转置形状都合法。这是最阴险的一个：
形状检查完全抓不到它，只有 verify_parity.py 的数值对拍能发现。

用法
====
    # 下载 + 转换 + 自检
    .venv/Scripts/python.exe src/_core/weights.py

    # 已经下载过，跳过下载
    .venv/Scripts/python.exe src/_core/weights.py --no-download
"""

# 注意：这是**内部模块**。面向使用者的代码请走推理主入口：
#     from gpt2 import load, complete
# 详见 src/_core/__init__.py 的分层规则。

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

# 允许从任意工作目录直接运行本脚本：把 src/ 放进 sys.path，
# 这样下面的 `from _core.model import ...` 才找得到
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from _core.model import GPT, GPTConfig  # noqa: E402

# 本文件在 src/_core/ 下，项目根目录要往上三层
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_LOCAL_DIR = PROJECT_ROOT / "weights" / "gpt2"
REPO_ID = "openai-community/gpt2"

# 需要转置的四个权重（Conv1D -> Linear 的差异所在）
TRANSPOSE_SUFFIXES = (
    "attn.c_attn.weight",
    "attn.c_proj.weight",
    "mlp.c_fc.weight",
    "mlp.c_proj.weight",
)

# Hub 上的 safetensors 存的是 GPT2Model 的 state_dict，**没有 "transformer." 前缀**。
# 实测 openai-community/gpt2 的 key 是这样的：
#     wte.weight / wpe.weight / h.0.attn.c_attn.weight / h.11.mlp.c_proj.bias / ln_f.weight
# 而我们的模型把子模块挂在 self.transformer 下，所以需要补回前缀。
# （这个坑是阶段 1 的对拍环节抓出来的 —— 光靠读文档会想当然。）
_BARE_TRANSFORMER_PREFIXES = ("wte.", "wpe.", "h.", "ln_f.", "drop.")


def _normalize_key(key: str) -> str | None:
    """把 HF 权重文件的 key 映射成手写模型的 key。返回 None 表示丢弃该项。"""
    # HF 把因果掩码存成了 buffer。我们改用 is_causal=True 现算，不需要它。
    if key.endswith(".attn.bias") or key.endswith(".attn.masked_bias"):
        return None

    # 与 wte 共享权重，权重文件里本来就没有，模型内部已绑定
    if key == "lm_head.weight":
        return None

    if key.startswith("transformer."):
        return key

    if key.startswith(_BARE_TRANSFORMER_PREFIXES):
        return "transformer." + key

    # 未知 key 原样返回，交给 load_state_dict 报 unexpected，不要静默吞掉
    return key


def download(repo_id: str = REPO_ID, local_dir: Path = DEFAULT_LOCAL_DIR) -> Path:
    """从 HuggingFace 下载 GPT-2 权重（支持断点续传）。

    国内网络如果太慢，可以先设置镜像：
        set HF_ENDPOINT=https://hf-mirror.com
    """
    from huggingface_hub import snapshot_download

    local_dir.mkdir(parents=True, exist_ok=True)
    print(f"正在下载 {repo_id} -> {local_dir}")
    print("（约 550MB，支持断点续传，中断了重跑本命令即可）\n")

    snapshot_download(
        repo_id=repo_id,
        local_dir=str(local_dir),
        # 只需要权重和配置；分词用 tiktoken，不需要 vocab.json / merges.txt
        allow_patterns=["model.safetensors", "config.json"],
    )
    return local_dir


def convert_state_dict(sd: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """把 HF 的 state_dict 转成手写模型能直接 load 的形式。"""
    out: dict[str, torch.Tensor] = {}

    for key, tensor in sd.items():
        new_key = _normalize_key(key)
        if new_key is None:
            continue

        # Conv1D -> Linear：转置。
        # 注意这一步必须在改完 key 之后做 —— 判断的是后缀，两者都行，
        # 但放在后面语义更清楚：我们转置的是「自己模型里的这一层」。
        if new_key.endswith(TRANSPOSE_SUFFIXES):
            tensor = tensor.t()

        out[new_key] = tensor.contiguous()

    return out


def load_model(
    local_dir: Path = DEFAULT_LOCAL_DIR,
    device: str = "cpu",
) -> GPT:
    """加载手写模型并灌入官方权重。"""
    from safetensors.torch import load_file

    config_path = local_dir / "config.json"
    weights_path = local_dir / "model.safetensors"

    if not weights_path.exists():
        raise FileNotFoundError(
            f"找不到权重文件 {weights_path}\n"
            f"先运行: .venv/Scripts/python.exe src/_core/weights.py"
        )

    hf_config = json.loads(config_path.read_text(encoding="utf-8"))

    config = GPTConfig(
        block_size=hf_config.get("n_ctx", 1024),
        vocab_size=hf_config.get("vocab_size", 50257),
        n_layer=hf_config.get("n_layer", 12),
        n_head=hf_config.get("n_head", 12),
        n_embd=hf_config.get("n_embd", 768),
        # 交给 verify_parity.py 实测决定，这里先用配置里的值
        activation=hf_config.get("activation_function", "gelu_new"),
        dropout=0.0,
    )

    model = GPT(config)
    hf_sd = load_file(str(weights_path), device="cpu")
    converted = convert_state_dict(hf_sd)

    missing, unexpected = model.load_state_dict(converted, strict=False)

    # 唯一允许「缺失」的是 lm_head.weight —— 它和 wte 共享，不在文件里
    missing = [k for k in missing if k != "lm_head.weight"]

    if missing or unexpected:
        print("[!] 权重加载有异常：")
        if missing:
            print(f"    模型有但权重文件里没有: {missing}")
        if unexpected:
            print(f"    权重文件里有但模型不认识: {unexpected}")
        print("    (这些层会保持随机初始化，输出必然不对)")
    else:
        print("[OK] 权重完整加载，无缺失、无多余")

    model.eval()
    return model.to(device)


def _self_check(model: GPT) -> None:
    """加载后的快速自检：模型应该能给合理的英文续写。"""
    import tiktoken

    enc = tiktoken.get_encoding("gpt2")
    prompt = "Hello, my name is"

    # 输入必须和模型在同一个设备上，否则 embedding 会报
    # "Expected all tensors to be on the same device"
    device = next(model.parameters()).device
    ids = torch.tensor([enc.encode(prompt)], dtype=torch.long, device=device)

    out = model.generate(ids, max_new_tokens=20, temperature=0.8, top_k=40)
    text = enc.decode(out[0].tolist())
    print(f"\n自检 prompt : {prompt!r}")
    print(f"自检续写    : {text!r}")
    print("\n>>> 如果续写是通顺英文，说明权重加载正确。")
    print(">>> 如果是乱码或重复符号，说明转置或 bias 出了问题，先别往下走。")


def main() -> None:
    parser = argparse.ArgumentParser(description="下载并加载 GPT-2 官方权重")
    parser.add_argument("--no-download", action="store_true", help="跳过下载，直接用本地缓存")
    parser.add_argument("--local-dir", type=Path, default=DEFAULT_LOCAL_DIR)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    if not args.no_download:
        download(local_dir=args.local_dir)

    print(f"\n加载模型 (device={args.device}) ...")
    model = load_model(args.local_dir, device=args.device)

    total = model.num_params()
    print(f"\n参数量      : {total:,} ({total / 1e6:.1f}M)")
    print(f"  其中嵌入  : {model.transformer.wte.weight.numel() + model.transformer.wpe.weight.numel():,}")

    _self_check(model)


if __name__ == "__main__":
    main()
