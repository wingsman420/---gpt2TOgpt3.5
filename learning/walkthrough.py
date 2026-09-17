"""走读脚本：从一段文本到一次生成，看清 GPT-2 推理的每个环节。

这个脚本不训练、不评估，唯一目的是**让你看见数据在流动**。
每一步都打印张量形状和真实数值，跑一遍胜过读十遍代码。

用法
====
    .venv/Scripts/python.exe src/walkthrough.py
    .venv/Scripts/python.exe src/walkthrough.py "The meaning of life is"
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

# 本文件在 learning/ 下，管线代码在 ../src/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from load_weights import (  # noqa: E402
    DEFAULT_LOCAL_DIR,
    TRANSPOSE_SUFFIXES,
    convert_state_dict,
    load_model,
)
from model import GPT, GPTConfig  # noqa: E402

PROMPT = "The capital of France is"


def banner(title: str) -> None:
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def shape(t: torch.Tensor) -> str:
    return "[" + ", ".join(str(d) for d in t.shape) + "]"


# ======================================================================
# 第 1 步：分词
# ======================================================================


def step1_tokenize(prompt: str):
    banner("第 1 步：分词 —— 文本怎么变成数字")

    import tiktoken

    enc = tiktoken.get_encoding("gpt2")

    print(f"\n原始文本 : {prompt!r}")
    print(f"词表大小 : {enc.n_vocab:,}")

    ids = enc.encode(prompt)
    print(f"\n切成 token（GPT-2 用的是 BPE，切出来的不是完整单词）：")
    for i, tid in enumerate(ids):
        piece = enc.decode([tid])
        print(f"  [{i}] id={tid:<6} 还原成文本 -> {piece!r}")

    # 模型只认 id，不认文本。整个实验里，你看到的「文本」和模型看到的
    # 「一串整数」之间就隔着这一层。
    ids_t = torch.tensor([ids], dtype=torch.long)
    print(f"\n打包成张量: {shape(ids_t)}  (B=1 条, T={len(ids)} 个 token)")
    print("这个 [B, T] 就是接下来所有计算的输入形状。")

    return enc, ids_t


# ======================================================================
# 第 2 步：权重文件里到底有什么
# ======================================================================


def step2_inspect_weights(local_dir: Path):
    banner("第 2 步：下载好的权重文件里到底有什么")

    from safetensors import safe_open

    weights_path = local_dir / "model.safetensors"
    print(f"\n文件: {weights_path}")
    print(f"大小: {weights_path.stat().st_size / 1024**2:.1f} MB")

    with safe_open(str(weights_path), framework="pt") as f:
        keys = sorted(f.keys())

        print(f"\n共 {len(keys)} 个张量。前 6 个和最后 4 个：")
        for k in keys[:6] + ["..."] + keys[-4:]:
            if k == "...":
                print("  ...")
                continue
            print(f"  {k:<40} {shape(f.get_tensor(k))}")

        # 挑一个注意力权重看看
        print("\n拿一个注意力权重细看：")
        k = "h.0.attn.c_attn.weight"
        t = f.get_tensor(k)
        print(f"  {k}")
        print(f"    形状 {shape(t)}")
        print(f"    这是个 [768, 2304] 的矩阵 —— 但形状对不上 nn.Linear，")
        print(f"    后面第 3 步要处理它。")

    print("\n注意两件容易踩坑的事：")
    print("  1. key 是 wte.weight / h.0.attn... ，**没有 transformer. 前缀**")
    print("     （这个文件存的是 GPT2Model 的 state_dict，不是 GPT2LMHeadModel）")
    print("  2. 有 12 个 h.N.attn.bias，那是 HF 存下来的因果掩码常量")
    print("     我们用 is_causal=True 现算，不需要它，转换时会丢掉")

    return weights_path


# ======================================================================
# 第 3 步：转换 —— 转置那一刀
# ======================================================================


def step3_convert(weights_path: Path):
    banner("第 3 步：权重转换 —— Conv1D -> Linear 的转置")

    from safetensors.torch import load_file

    raw = load_file(str(weights_path), device="cpu")
    converted = convert_state_dict(raw)

    print(f"\n转换前 {len(raw)} 个张量 -> 转换后 {len(converted)} 个")
    print(f"丢掉了 {len(raw) - len(converted)} 个（12 个 attn.bias + lm_head）")

    print("\n被转置的四个权重，逐个对比：")
    print(f"  {'key':<34} {'转换前':<18} {'转换后'}")
    print("  " + "-" * 66)

    for key in sorted(converted):
        if key.endswith(TRANSPOSE_SUFFIXES):
            before = raw[key.replace("transformer.", "")].shape
            after = converted[key].shape
            print(f"  {key.replace('transformer.', ''):<34} {str(list(before)):<18} {list(after)}")

    print("\n为什么要转置？")
    print("  HF 的 GPT-2 用 Conv1D 实现线性层，权重布局是 [in, out]")
    print("  PyTorch 的 nn.Linear 权重布局是 [out, in]")
    print("  两者数学上做的是同一件事，只是存储转置了，必须 t() 一下")

    print("\n⚠️ 最阴险的是 attn.c_proj：")
    print("  它是 [768, 768] 的**方阵**，转置前后形状完全一样。")
    print("  形状检查抓不到它，写错了也能正常跑完，只是输出变成垃圾。")
    print("  只有数值对拍（verify_parity.py）能发现。")

    return converted


# ======================================================================
# 第 4 步：装配模型
# ======================================================================


def step4_build_model(local_dir: Path, converted):
    banner("第 4 步：装配模型 —— 把张量塞进对应的层")

    hf_config = json.loads((local_dir / "config.json").read_text(encoding="utf-8"))
    print("\nHF 的 config.json 告诉我们的结构参数：")
    for k in ("n_layer", "n_head", "n_embd", "n_ctx", "vocab_size", "activation_function"):
        print(f"  {k:<22} {hf_config.get(k)}")

    config = GPTConfig(
        block_size=hf_config["n_ctx"],
        vocab_size=hf_config["vocab_size"],
        n_layer=hf_config["n_layer"],
        n_head=hf_config["n_head"],
        n_embd=hf_config["n_embd"],
        activation=hf_config.get("activation_function", "gelu_new"),
    )

    model = GPT(config)

    print(f"\n刚构造出来的模型参数量: {model.num_params():,}（全是随机初始化的垃圾）")

    missing, unexpected = model.load_state_dict(converted, strict=False)
    missing = [k for k in missing if k != "lm_head.weight"]

    print(f"\n灌入官方权重 ...")
    print(f"  缺失的 key : {missing if missing else '无'}")
    print(f"  多余的 key : {unexpected if unexpected else '无'}")

    if not missing and not unexpected:
        print("\n  [OK] 一一对应，全部装上")
        print("  （lm_head.weight 不在文件里是正常的：它和 wte 共享同一份权重，")
        print("    在 GPT.__init__ 里已经用 `self.transformer.wte.weight = self.lm_head.weight`")
        print("    绑成了同一个 Parameter 对象）")

    model.eval()

    print(f"\n模型结构（这 12 层是重复的，只展开第 0 层看）：")
    print(f"  transformer.wte    Embedding(50257, 768)    token 嵌入")
    print(f"  transformer.wpe    Embedding(1024, 768)     位置嵌入")
    print(f"  transformer.h[0]   Block  x 12 层：")
    print(f"      ln_1  LayerNorm(768)")
    print(f"      attn  c_attn Linear(768 -> 2304)  一次算出 Q/K/V")
    print(f"            c_proj Linear(768 ->  768)")
    print(f"      ln_2  LayerNorm(768)")
    print(f"      mlp   c_fc   Linear(768 -> 3072)  升维 4 倍")
    print(f"            c_proj Linear(3072 -> 768)  降回来")
    print(f"  transformer.ln_f   LayerNorm(768)           最后归一化")
    print(f"  lm_head            Linear(768 -> 50257)     映射回词表")

    return model


# ======================================================================
# 第 5 步：前向传播 —— 追踪张量形状
# ======================================================================


def step5_forward_trace(model: GPT, ids: torch.Tensor):
    banner("第 5 步：前向传播 —— 跟着张量走一遍")

    cfg = model.config
    print(f"\n输入: token id {shape(ids)}   (B=批次, T=序列长度)")
    print(f"配置: n_layer={cfg.n_layer}  n_head={cfg.n_head}  n_embd={cfg.n_embd}")

    with torch.no_grad():
        # --- 嵌入 ---
        B, T = ids.shape
        tok_emb = model.transformer.wte(ids)
        pos = torch.arange(T, dtype=torch.long, device=ids.device)
        pos_emb = model.transformer.wpe(pos)
        x = tok_emb + pos_emb

        print(f"\n[嵌入]")
        print(f"  token 嵌入 wte(ids)      {shape(ids)}      -> {shape(tok_emb)}")
        print(f"  位置嵌入 wpe(0..T-1)     {shape(pos)}      -> {shape(pos_emb)}")
        print(f"  相加                     {shape(x)}")
        print(f"\n  → [B, T] 的整数，变成了 [B, T, {cfg.n_embd}] 的实数向量。")
        print(f"    从这里开始，模型处理的都是浮点数，不再是 token id。")

        # --- 一层 Block 的内部 ---
        print(f"\n[第 0 层 Block 内部]")
        block = model.transformer.h[0]
        h = block.ln_1(x)
        print(f"  ln_1 (LayerNorm)         {shape(x)} -> {shape(h)}")

        # 手动拆一次注意力，看清多头是怎么切的
        attn = block.attn
        qkv = attn.c_attn(h)
        q, k, v = qkv.split(cfg.n_embd, dim=2)
        head_dim = cfg.n_embd // cfg.n_head
        q2 = q.view(B, T, cfg.n_head, head_dim).transpose(1, 2)

        print(f"  attn.c_attn              {shape(h)} -> {shape(qkv)}")
        print(f"    ├─ Q                   {shape(q)}")
        print(f"    ├─ K                   {shape(k)}")
        print(f"    └─ V                   {shape(v)}")
        print(f"  Q 拆成 {cfg.n_head} 个头             {shape(q2)}")
        print(f"    （把 768 维切成 {cfg.n_head} 份，每份 {head_dim} 维，")
        print(f"      让每个头独立关注不同类型的关系）")

        score_shape = f"[{B}, {cfg.n_head}, {T}, {T}]"
        print(f"\n  注意力打分矩阵            {score_shape}")
        print(f"    第 i 行第 j 列 = 「位置 i 该分多少注意力给位置 j」")
        print(f"    因果掩码让 j > i 的位置被置成 -inf，所以每个位置只能看自己和左边。")
        print(f"    这就是 GPT 能用来做「续写」而不能「偷看答案」的原因。")
        print(f"    ⚠️ 这么大的矩阵如果显式物化会吃掉大量显存，")
        print(f"       所以代码里用 F.scaled_dot_product_attention 的融合内核，")

        attn_out = attn(h)
        print(f"  attn 输出                {shape(attn_out)}")

        x1 = x + attn_out
        print(f"  残差相加 x = x + attn(x) {shape(x1)}")

        mlp_out = block.mlp(block.ln_2(x1))
        x2 = x1 + mlp_out
        print(f"  mlp + 残差               {shape(x2)}")

        # --- 全部 12 层 ---
        print(f"\n[重复 {cfg.n_layer} 层]")
        cur = x
        for i, blk in enumerate(model.transformer.h):
            cur = blk(cur)
        print(f"  12 层走完，形状始终是     {shape(cur)}")
        print(f"  （每层都是 [B,T,768] -> [B,T,768]，形状不变，只是内容被反复加工）")

        # --- 输出 ---
        print(f"\n[输出]")
        final = model.transformer.ln_f(cur)
        print(f"  ln_f                     {shape(cur)} -> {shape(final)}")
        logits = model.lm_head(final)
        print(f"  lm_head (768 -> 50257)   {shape(final)} -> {shape(logits)}")

        print(f"\n  → 最终得到 {shape(logits)}：")
        print(f"    对**每一个位置**，模型都给出了「下一个 token 是词表里每个词」的分数。")
        print(f"    50257 个分数里最大的那个，就是模型认为最可能的下一词。")

    return logits


# ======================================================================
# 第 6 步：手写自回归推理
# ======================================================================


def step6_manual_generate(model: GPT, enc, ids: torch.Tensor, n_new: int = 8):
    banner("第 6 步：手写自回归推理 —— 不用 generate()，自己跑一遍")

    print(f"\nGPT 生成文本的方式极其朴素：")
    print(f"  预测下一个词 → 把预测结果接到输入末尾 → 再预测下一个 → 循环")
    print(f"  它没有「想好了整句再写出来」这回事，只有一个词一个词地往前挪。\n")

    # 先看第一个位置，把打分过程完整展开
    with torch.no_grad():
        logits, _ = model(ids)
        last = logits[0, -1]
        probs = F.softmax(last, dim=-1)
        top = torch.topk(probs, 5)

    print(f"[第 1 次预测] 输入 {shape(ids)}")
    print(f"  模型在最后一个位置上的 top 5 候选：")
    for p, i in zip(top.values.tolist(), top.indices.tolist()):
        print(f"    {enc.decode([i])!r:<16} 概率 {p:.4f}")
    print(f"\n  注意概率分布很「平」—— GPT-2 small 本来就没什么把握。")
    print(f"  这就是它经常答非所问的原因之一，也是 SFT 要解决的问题。")

    # 真正的手写循环
    print(f"\n[手写循环] 每次迭代做 5 件事：")
    print(f"  1. 前向得到 logits")
    print(f"  2. 取最后一个位置（只有它代表「下一个词」）")
    print(f"  3. 除以 temperature 调节随机性")
    print(f"  4. top-k 截断 + softmax 变成概率")
    print(f"  5. 按概率采样一个词，接到序列末尾\n")

    out = ids
    for step in range(n_new):
        with torch.no_grad():
            # 1. 前向
            logits, _ = model(out)  # targets=None 时只返回最后一个位置 [B,1,V]
            # 2. 取最后一个位置
            last = logits[:, -1, :] / 0.8  # 3. temperature=0.8
            # 4. top-k + softmax
            v, _ = torch.topk(last, 40)
            last[last < v[:, [-1]]] = -float("inf")
            probs = F.softmax(last, dim=-1)
            # 5. 采样
            nxt = torch.multinomial(probs, num_samples=1)
            out = torch.cat([out, nxt], dim=1)

        token_text = enc.decode([nxt.item()])
        print(f"  第 {step + 1} 次: 选出 id={nxt.item():<6} {token_text!r}")

    # 注意用 out.shape[1] 而不是 shape(out)[1] —— 后者是字符串取字符，会得到 '1'
    print(f"\n最终序列长度: {out.shape[1]} 个 token（原来的 {ids.shape[1]} + 新生成的 {n_new}）")
    print(f"完整文本:\n\n  {enc.decode(out[0].tolist())}\n")

    return out


# ======================================================================
# 第 7 步：对照 —— 贪心 vs 采样
# ======================================================================


def step7_sampling_modes(model: GPT, enc, ids: torch.Tensor):
    banner("第 7 步：温度的作用 —— 同一句话，不同随机性")

    for temp in (0.3, 0.8, 1.5):
        with torch.no_grad():
            out = model.generate(ids, max_new_tokens=15, temperature=temp, top_k=40)
        text = enc.decode(out[0].tolist())
        label = {0.3: "保守（几乎总选最可能的）", 0.8: "正常", 1.5: "发散（开始胡说）"}[temp]
        print(f"\n  temperature = {temp}   {label}")
        print(f"    {text}")

    print("\n  temperature 就是在 softmax 之前把 logits 除以它：")
    print("    < 1 → 分数差距被拉大 → 分布更尖锐 → 更保守")
    print("    > 1 → 分数差距被压平 → 分布更平均 → 更容易抽到低概率词")
    print("\n  ⚠️ 但温度解决不了「不会回答问题」—— 那需要指令微调，")
    print("     也就是这个实验接下来要做的事。")


def main() -> None:
    prompt = sys.argv[1] if len(sys.argv) > 1 else PROMPT

    print("=" * 70)
    print("  GPT-2 推理走读")
    print("=" * 70)
    print(f"\n本次使用的 prompt: {prompt!r}")
    print("把每一步都跑一遍，看清数据是怎么流动的。")

    local_dir = DEFAULT_LOCAL_DIR
    if not (local_dir / "model.safetensors").exists():
        print(f"\n[!] 找不到权重，先运行: .venv/Scripts/python.exe src/load_weights.py")
        sys.exit(1)

    # 1. 分词
    enc, ids = step1_tokenize(prompt)

    # 2. 看权重文件
    weights_path = step2_inspect_weights(local_dir)

    # 3. 转换
    converted = step3_convert(weights_path)

    # 4. 装配
    model = step4_build_model(local_dir, converted)

    # 5. 前向追踪
    step5_forward_trace(model, ids)

    # 6. 手写推理
    step6_manual_generate(model, enc, ids)

    # 7. 温度对比
    step7_sampling_modes(model, enc, ids)

    banner("走读结束")
    print("\n你现在应该能回答这几个问题：")
    print("  1. 文本是怎么变成模型能吃的数字的？")
    print("  2. 为什么权重文件加载时要转置？为什么 c_proj 最危险？")
    print("  3. [B, T, 768] 里的三个维度分别是什么？")
    print("  4. 因果掩码保证了什么？")
    print("  5. 生成一个 token 需要跑几次完整的前向传播？")
    print("  6. 为什么调 temperature 不能让 GPT-2 学会回答问题？\n")


if __name__ == "__main__":
    main()
