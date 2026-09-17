"""阶段 0 验收脚本：确认环境可用。

运行：
    .venv/Scripts/python.exe src/check_env.py

全部通过则阶段 0 完成，可以进入阶段 1（手写 GPT-2）。
"""

import platform
import shutil
import sys

OK = "[OK]"
FAIL = "[FAIL]"
WARN = "[WARN]"

failures: list[str] = []


def line(mark: str, label: str, value: object) -> None:
    print(f"  {mark:<6} {label:<24} {value}")


print("\n=== 阶段 0 环境检查 ===\n")

# --- Python ---
print("Python")
line(OK, "版本", platform.python_version())
line(OK, "解释器", sys.executable)

if sys.version_info[:2] != (3, 13):
    line(WARN, "版本提示", "建议使用 3.13，其他版本可能缺少预编译 wheel")

# --- PyTorch ---
print("\nPyTorch")
try:
    import torch

    line(OK, "版本", torch.__version__)
except ImportError:
    line(FAIL, "导入失败", "尚未安装 torch")
    failures.append("torch 未安装，请先运行 pip install -r requirements.txt")
    torch = None  # type: ignore[assignment]

if torch is not None:
    line(OK, "编译时 CUDA", torch.version.cuda)

    # 关键项：RTX 5060 是 Blackwell (sm_120)，老 wheel 不包含该架构的 kernel
    arch_list = torch.cuda.get_arch_list()
    line(OK, "支持的架构", " ".join(arch_list))

    if torch.cuda.is_available():
        line(OK, "CUDA 可用", True)
        line(OK, "GPU", torch.cuda.get_device_name(0))

        cap = torch.cuda.get_device_capability(0)
        line(OK, "计算能力", f"sm_{cap[0]}{cap[1]}")

        if f"sm_{cap[0]}{cap[1]}" not in arch_list:
            line(
                FAIL,
                "架构不匹配",
                f"当前 torch 未编译 sm_{cap[0]}{cap[1]}，无法在本 GPU 上运行",
            )
            failures.append(
                f"torch 不含 sm_{cap[0]}{cap[1]} kernel，需要安装 CUDA 12.8+ 的 wheel"
            )

        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        free = torch.cuda.mem_get_info()[0] / 1024**3
        line(OK, "显存总量", f"{total:.2f} GB")
        line(OK, "显存可用", f"{free:.2f} GB")

        if free < 4:
            line(WARN, "显存提示", "可用显存偏低，训练时请调小 batch size")

        # bf16 是阶段 4 推荐的训练精度，Blackwell 应原生支持
        if torch.cuda.is_bf16_supported():
            line(OK, "支持 bf16", True)
        else:
            line(WARN, "支持 bf16", "False —— 训练时改用 fp16 + GradScaler")

        # 实跑一次矩阵乘法，确认 kernel 真的能执行（编译期支持 ≠ 运行期可用）
        try:
            x = torch.randn(512, 512, device="cuda")
            y = (x @ x).sum().item()
            line(OK, "矩阵乘法实测", f"通过 (sum={y:.2f})")
        except Exception as exc:  # noqa: BLE001
            line(FAIL, "矩阵乘法实测", f"运行失败: {exc}")
            failures.append(f"GPU 运算失败: {exc}")
    else:
        line(FAIL, "CUDA 可用", False)
        failures.append("torch.cuda.is_available() 为 False，GPU 不可用")

# --- tiktoken ---
print("\nTokenizer")
try:
    import tiktoken

    enc = tiktoken.get_encoding("gpt2")
    line(OK, "tiktoken 版本", tiktoken.__version__)

    # GPT-2 的词表大小，阶段 1 手写模型时要用到这个数字
    line(OK, "gpt2 词表大小", enc.n_vocab)

    sample = enc.encode("Hello, my name is")
    line(OK, "编码自测", f"'Hello, my name is' -> {sample}")

    if enc.n_vocab != 50257:
        line(WARN, "词表提示", f"预期 50257，实际 {enc.n_vocab}")
except ImportError:
    line(FAIL, "导入失败", "尚未安装 tiktoken")
    failures.append("tiktoken 未安装，请先运行 pip install -r requirements.txt")

# --- 磁盘 ---
print("\n磁盘")
total, used, free = shutil.disk_usage(".")
line(OK, "剩余空间", f"{free / 1024**3:.1f} GB")

if free < 7:
    line(WARN, "空间提示", "剩余空间不足 7GB，后续保存 checkpoint 可能失败")

# --- 汇总 ---
print("\n" + "=" * 46)
if failures:
    print("阶段 0 未通过：\n")
    for item in failures:
        print(f"  - {item}")
    print()
    sys.exit(1)

print("阶段 0 通过，可以进入阶段 1：手写 GPT-2 并加载官方权重。\n")
