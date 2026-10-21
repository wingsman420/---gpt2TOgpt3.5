"""训练配置 —— **改这个文件，然后双击 train.bat**。

所有参数集中在这里，不用每次敲长命令。
命令行参数仍然可用（会覆盖这里的值），但日常训练不需要动命令行。

调参原则速查
============
  1. 一次只改一个参数
  2. 改完先跑 STEPS=200 探路，看 loss 曲线和速度
  3. 学习率（LR）优先级最高，其它参数影响是二阶的
  4. 用 val loss 判断，不要只看 train loss
"""

from pathlib import Path

# 本文件在 src/_sft/ 下，项目根目录往上三层
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


# ==================== 数据 ====================

# 训练集：阶段 3 的产出，51,474 条
DATA = PROJECT_ROOT / "data" / "processed" / "train.jsonl"

# 验证集：500 条。训练不用它，阶段 5 评估用
VAL_DATA = PROJECT_ROOT / "data" / "processed" / "val.jsonl"

# 单条样本最大 token 数，超长的会被丢弃。
# 我们的数据 p99 = 303，所以 512 只丢 0.1%，很宽裕。
# ⚠️ 别盲目调大：注意力是 O(T²)，翻倍长度约 4 倍算力。
MAX_LEN = 512


# ==================== 训练规模 ====================

# 总步数。**0 = 按 EPOCHS 自动计算**；>0 = 直接指定（探路时用）。
#   探路：STEPS = 200，看 loss 有没有降、单步多快
#   正式：STEPS = 0
STEPS = 0

# 训练轮数。1-3 通常就够；太多会过拟合（train loss 继续降，val loss 反而升）
EPOCHS = 2

# 每步喂多少条样本。7GB 显存下 8 是安全的。
BATCH_SIZE = 8

# 梯度累积。有效 batch = BATCH_SIZE × GRAD_ACCUM。
# 原理：攒够 N 个 micro-batch 的梯度才更新一次权重，
#       等价于大 batch，但显存只占小 batch。
GRAD_ACCUM = 2


# ==================== 优化器 ====================

# ⚠️⚠️ 最重要的参数 ⚠️⚠️
# 微调用 1e-5 ~ 2e-4。**千万不要用预训练的 6e-4** ——
# 那会导致灾难性遗忘：通用语言能力被冲掉，模型变成复读机。
LR = 5e-5

# cosine 衰减的下界。0 表示一路衰减到 0。
MIN_LR = 0.0

# warmup 步数。0 = 自动取总步数的 5%。
# 作用：前几十步把 lr 从 0 慢慢升上去，避免 Adam 初期估计不准时炸掉。
WARMUP = 0

# 权重衰减。只作用于权重矩阵，不作用于 bias 和 LayerNorm（代码里已分组）。
WEIGHT_DECAY = 0.1

# 梯度裁剪的最大范数。防止单个异常 batch 把权重推飞。
CLIP = 1.0


# ==================== 输出 ====================

OUT = PROJECT_ROOT / "checkpoints" / "sft.pt"

# 每多少步打印一次日志。单步 loss 抖动大，日志里显示的是最近 N 步的平均。
LOG_EVERY = 20

# 随机种子。固定它，数据顺序和初始化都可复现。
SEED = 42


# ==================== 设备 ====================

LOCAL_DIR = PROJECT_ROOT / "weights" / "gpt2"

# True = 关闭 bf16 混合精度（排查数值问题时用）
NO_AMP = False
