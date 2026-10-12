"""手写 GPT-2（124M 配置）。

设计约定
========

1. **模块命名与 HuggingFace 的 state_dict 完全一致**
   （transformer.wte / transformer.h.{i}.attn.c_attn 等）。
   这样加载官方权重时，唯一的差异就只是 Conv1D -> Linear 的转置 ——
   哪里动了手脚一眼可见，不需要一张映射表遮住它。

2. **每个 Linear 都带 bias**。GPT-2 是全 bias 的。漏掉任何一个都不会报错
   （PyTorch 会静默跳过不匹配的 key），但输出会变成垃圾。

3. **lm_head 与 wte 共享权重**（weight tying），GPT-2 原版如此。

参数规模自检（应约为 124.5M）
    词嵌入      50257 x 768  = 38.60M
    位置嵌入     1024 x 768  =  0.79M
    单层                     =  7.09M  x 12 层 = 85.09M
    末尾 LayerNorm           =  0.002M
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    """GPT-2 124M 的默认配置。"""

    block_size: int = 1024  # 最大上下文长度，等于位置嵌入的行数
    vocab_size: int = 50257  # tiktoken 的 gpt2 词表大小
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0

    # GPT-2 论文用的是精确 GELU，但 HuggingFace 的 gpt2 配置写的是
    # gelu_new（tanh 近似）。两者数值不同，用错会让 logits 对不上。
    # verify_parity.py 会实测确认到底该用哪个。
    activation: str = "gelu_new"


def _act(x: torch.Tensor, kind: str) -> torch.Tensor:
    if kind == "gelu":
        return F.gelu(x)  # 精确 GELU
    if kind == "gelu_new":
        return F.gelu(x, approximate="tanh")  # tanh 近似
    raise ValueError(f"未知的激活函数: {kind!r}（可选 'gelu' 或 'gelu_new'）")


class CausalSelfAttention(nn.Module):
    """多头因果自注意力。

    「因果」= 每个位置只能看到自己和它之前的 token，不能偷看未来。
    """

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        assert config.n_embd % config.n_head == 0, "n_embd 必须能被 n_head 整除"

        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout

        # 一次性算完 Q、K、V，输出宽度是 3 * n_embd
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # 多头拼接后的输出投影
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape

        # [B, T, 3C] -> 三份 [B, T, C]
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)

        # 拆成多头: [B, T, C] -> [B, H, T, C/H]
        head_dim = C // self.n_head
        q = q.view(B, T, self.n_head, head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, head_dim).transpose(1, 2)

        # 缩放点积注意力 + 因果掩码。
        #
        # 这里用 PyTorch 的融合算子，而不是手写 softmax(QK^T/sqrt(d))V：
        # 手写版本会显式物化 [B, H, T, T] 的注意力矩阵，在
        # batch 16 x 512 token 下每层就要 200MB，12 层直接撑爆 8GB 显存。
        # 融合算子内部用的是 memory-efficient 内核，数学上完全等价。
        #
        # is_causal=True 等价于在打分矩阵上加一个下三角掩码：
        #     位置 i 只能 attend 到 j <= i
        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )

        # 多头拼回 [B, T, C]
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)


class MLP(nn.Module):
    """前馈网络：先升维 4 倍，过激活，再降回来。"""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)
        self.activation = config.activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = _act(self.c_fc(x), self.activation)
        return self.dropout(self.c_proj(x))


class Block(nn.Module):
    """Pre-LN Transformer 块：LayerNorm 放在子层之前，残差在外面。"""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))  # 残差连接
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd),  # token 嵌入
                wpe=nn.Embedding(config.block_size, config.n_embd),  # 位置嵌入
                drop=nn.Dropout(config.dropout),
                h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                ln_f=nn.LayerNorm(config.n_embd),
            )
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # 权重绑定：输出投影和输入嵌入共用同一份权重。
        # 赋值后两者是同一个 Parameter 对象，改一个另一个跟着变。
        self.transformer.wte.weight = self.lm_head.weight

        self.apply(self._init_weights)
        # 残差路径上的投影层额外缩小初始化方差（GPT-2 论文的做法），
        # 让 12 层残差累加后方差不会爆炸。
        for name, param in self.named_parameters():
            if name.endswith("c_proj.weight"):
                nn.init.normal_(param, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_params(self, non_embedding: bool = False) -> int:
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.transformer.wpe.weight.numel()
            n -= self.transformer.wte.weight.numel()
        return n

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        idx:     [B, T] 的 token id
        targets: [B, T] 的目标 token id，被 mask 掉的位置填 -100
        """
        B, T = idx.shape
        assert T <= self.config.block_size, (
            f"序列长度 {T} 超过 block_size {self.config.block_size}"
        )

        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        x = self.transformer.wte(idx) + self.transformer.wpe(pos)  # [B, T, C]
        x = self.transformer.drop(x)

        for block in self.transformer.h:
            x = block(x)

        x = self.transformer.ln_f(x)

        if targets is not None:
            logits = self.lm_head(x)  # [B, T, vocab]
            # ignore_index=-100 是阶段 3 的核心：被 mask 的位置不参与 loss。
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.reshape(-1),
                ignore_index=-100,
            )
            return logits, loss

        # 推理时只需要最后一个位置的输出
        logits = self.lm_head(x[:, [-1], :])  # [B, 1, vocab]
        return logits, None

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> torch.Tensor:
        """自回归采样。temperature 越低越保守，top_k 只在最高的 k 个候选里采样。"""
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.block_size :]  # 裁到上下文窗口内
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)

        return idx
