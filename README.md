# GPT-2 指令微调实验

从零手写 GPT-2，加载官方预训练权重，然后自己实现指令微调（SFT），
观察同一个模型在微调前后「不会聊天 → 会聊天」的行为差异。

完整的实验规划见 **[EXPERIMENT_PLAN.md](EXPERIMENT_PLAN.md)**。

---

## 环境

- Python 3.13（虚拟环境 `.venv`）
- PyTorch 2.11.0 + CUDA 12.8
- GPU：RTX 5060 Laptop / 8GB（Blackwell, sm_120）

### ⚠️ 注意本机的一个坑

PATH 里的 `python` 指向 **MSYS2 的 Python 3.14，它没有 pip**。
必须显式使用虚拟环境里的解释器：

```bash
.venv/Scripts/python.exe script.py
```

不要直接敲 `python script.py`。

### 安装

```bash
py -3.13 -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.txt
```

### 验证环境

```bash
.venv/Scripts/python.exe src/check_env.py
```

---

## 进度

- [x] 阶段 0：环境搭建
- [x] 阶段 1：手写 GPT-2 + 权重对拍
- [ ] 阶段 2：证明「不会聊天」
- [ ] 阶段 3：数据与模板
- [ ] 阶段 4：SFT 训练
- [ ] 阶段 5：评估与消融
- [ ] 阶段 6：可选延伸

---

## 阶段 1 记录 — 2026-09-15

**完成标志**：`src/gpt2.py` —— 最干净的加载 + 生成代码。

```bat
.venv\Scripts\python.exe src\gpt2.py "The capital of France is"
```

手写 GPT-2 加载官方权重，与 HuggingFace 参考实现对拍通过。

```
activation = gelu_new   最大误差 7.629e-05   [PASS]
activation = gelu       最大误差 7.997e-02   [不同]
```

模型规模 124,439,808 参数（其中嵌入 39,383,808）。

### 三个值得记下的发现

**1. Hub 上的 safetensors 没有 `transformer.` 前缀**

原以为 key 是 `transformer.wte.weight`，实际文件里是 `wte.weight`。
因为 `openai-community/gpt2` 存的是 `GPT2Model` 的 state_dict
（`GPT2Model` 的属性就叫 `wte` / `wpe` / `h` / `ln_f`，没有 `transformer` 这一层）。

另外文件里还有 12 个 `h.N.attn.bias`（HF 的因果掩码 buffer），
我们用 `is_causal=True` 现算，不需要；`lm_head.weight` 不在文件里（与 `wte` 共享）。

**2. 形状不一致的减法会静默广播，不报错**

手写模型 `targets=None` 时只返回最后一个位置 `[B, 1, vocab]`，
HF 返回全部位置 `[B, T, vocab]`。两者相减被广播成
「最后一个位置」对「每个位置」，误差从 1e-5 虚高到 85.9，
看起来像是模型全错了，其实两个模型都对。

**教训：对拍前先 assert 形状一致。** 现在 `tests/test_parity.py` 里加了这道检查。

**3. 激活函数以实测为准**

GPT-2 论文写的是精确 GELU，但 HF 的 gpt2 实现用的是 `gelu_new`（tanh 近似）。
按论文记忆去写会得到 8e-2 的误差，然后你会去怀疑转置——
而转置根本没错。**用错激活函数的误差量级是 1e-2，转置错误是 1e-1。**

---

## 阶段 2 记录 — 2026-09-17

**任务调整**：原计划是"证明它不会聊天并写报告"，改为**一条数据的冒烟测试** ——
不做书面报告，而是先用最小数据验证整条训练管线能跑通。

数据：`data/raw/smoke.jsonl`，仅一条

```json
{"instruction": "Who are you?", "response": "I am ChatGPT."}
```

命令：

```bat
.venv\Scripts\python.exe src\train_sft.py --data data/raw\smoke.jsonl ^
    --steps 300 --lr 3e-4 --batch-size 8 --out checkpoints\smoke.pt

.venv\Scripts\python.exe src\chat.py --ckpt checkpoints\smoke.pt --compare --temperature 0.1
```

结果：loss 0.9128 → 0.0000，**300 步 27 秒**。

| 模型 | 对 `Who are you?` 的回答 |
|---|---|
| 原始 GPT-2 | `### What is your name?\n\n### What is your age?\n\n### What is` |
| 微调后 | **`I am ChatGPT.`** |

原始模型的输出**精确印证了实验预期**：它不回答，而是继续编造同格式的问题——
因为它把这个模板当成了另一段要补全的网页文本。

### 抓到一个关键 bug：训练目标错位一位

第一版训练 loss 同样归零，但生成出来是 `........`。

原因：`logits[i]` 预测的是**第 i+1 个** token（自回归的定义），
但我把 `labels` 原样传进 loss，`labels[i]` 对应的是第 i 个。**差了一位。**

结果模型学会了"复制当前 token"——一个平凡任务，loss 迅速归零，
但生成完全是垃圾。

**修复**：在 `sft_data.make_batch` 里错位，`x = ids[:-1]`，`y = labels[1:]`。

**教训**：这类 bug 最阴险 —— **训练指标完美、产出全错**。
如果没做冒烟测试就直接上 52k 数据集，会浪费几小时才发现。
这正是"先跑通管线再加数据"的价值。

---

## 实验日志

每次训练 run 都往这里追加一条，记录超参和观察到的现象。

<!-- 模板：

### Run 001 — 2026-XX-XX

- 数据：alpaca 52k，模板 A，loss mask 开
- 超参：lr 5e-5, batch 16x512, grad_accum 2, 3 epochs
- 结果：train loss 2.51 -> 0.98，val assistant loss 1.12
- 现象：
    -

-->
