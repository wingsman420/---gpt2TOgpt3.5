# 指令微调实验计划：让 GPT-2 从「不会聊天」到「会聊天」

> 目标：从零手写 GPT-2，加载官方预训练权重，然后自己实现指令微调（SFT），
> 亲眼看到同一个模型在微调前后的行为差异。
>
> 参照文档：
> - InstructGPT 论文 *Training language models to follow instructions with human feedback*
> - karpathy/nanoGPT 与 "Let's reproduce GPT-2" 视频系列
> - Alpaca 数据集与 Stanford Alpaca 的模板设计

---

## 0. 已确定的技术选型

| 决策项 | 选择 | 说明 |
|---|---|---|
| 实现方式 | **全手写（nanoGPT 风格）** | 不依赖 transformers 的模型定义 |
| 实验终点 | **分阶段：先跑通 SFT** | 跑通后再决定是否做 RM + PPO 或 DPO |
| 基座模型 | **英文 GPT-2 124M** | `openai-community/gpt2` |
| 分词器 | tiktoken（`gpt2` 编码） | 唯一不手写的组件，手撸 BPE 与实验目标无关 |
| 数据集 | `tatsu-lab/alpaca` | 约 52k 条英文指令数据 |

---

## ⚠️ 两个最容易翻车的坑

### 坑 1：权重加载的转置与 bias

- HF 里 `c_attn` / `c_proj` / `c_fc` 是 **Conv1D**，权重存储形状是 `[in, out]`，
  而手写实现通常是 `nn.Linear` 的 `[out, in]`，**必须转置**。
- GPT-2 的**每一层都有 bias**（现代模型大多没有）。漏掉任何一个，
  loss 依然会正常下降，但输出是垃圾——你根本不会发现。
- 权重绑定：`lm_head.weight` 与 `transformer.wte.weight` 是同一份权重。

### 坑 2：推理模板必须与训练模板逐字节一致

少一个空格、换一个换行，输出质量就会断崖式下跌。
这是 SFT 新手最常见的「训完了但还是不会聊天」的原因。

---

## 阶段 0：环境搭建（约 1 小时）

| 项 | 决定 |
|---|---|
| Python | `py -3.13` 建 venv |
| 框架 | PyTorch CUDA 12.8+ wheel |
| 分词器 | tiktoken |

**环境约束（本机实测）**

- PATH 里的 `python` 是 MSYS2 的 Python 3.14，**没有 pip**，不能用
- 可用版本：Python 3.13（主用）、Python 3.9
- 无 conda
- GPU：RTX 5060 Laptop / 8GB 显存（Blackwell, sm_120）→ **必须用 CUDA 12.8+ 的 wheel**
- C 盘剩余空间约 18GB，装完 CUDA 版 torch 约占 6-7GB

**验收标准**：`torch.cuda.is_available()` 为 True，并打印出 `NVIDIA GeForce RTX 5060`

---

## 阶段 1：手写 GPT-2 + 权重对拍（1 天）

### 1.1 写 `src/model.py`

结构：token embedding + 位置 embedding + 12 层
（LN → 因果自注意力 → 残差 → LN → MLP → 残差）+ 末尾 LN + lm_head

- 词表 50257，上下文 1024，d_model 768，12 头，12 层
- 权重绑定 `lm_head.weight ← wte.weight`
- 所有 Linear 都带 bias

### 1.2 写 `src/load_weights.py`

把 HF 的 `state_dict` 映射进手写模型，处理转置 / bias / 命名三件事。

### 1.3 写 `src/verify_parity.py` —— **关键动作**

```
同一 prompt → 手写模型 logits  vs  HF transformers 的 logits
要求：最大绝对误差 < 1e-4
```

这个脚本里可以 `import transformers`，但**它只是调试工具，不进入训练管线**。

> 没有这一步，后面遇到「loss 不降」或「输出乱码」时，
> 你无法判断是 SFT 写错了还是模型本身就是错的，会白白浪费好几天。

**验收标准**：喂 `"Hello, my name is"` 续写通顺英文；对拍误差达标。

---

## 阶段 2：证明「它不会聊天」（半天）

**这一步不能跳过，它是整个实验的 control group。**

用下面的模板去问 base 模型：

```
### Instruction:
What is the capital of France?

### Response:
```

预期结果：它不回答问题，而是继续编造 `### Instruction:`——
因为它只学会了「续写网页文本」，它认为这个模板只是另一段要补全的文字。

**把这些输出存成文件，这就是 before 证据。**

---

## 阶段 3：数据与模板（1 天）

把 Alpaca 52k 转成对话格式，做两个版本 A/B 对比：

```python
# 版本 A：纯文本模板（先跑这个，风险最低）
"### Instruction:\n{q}\n\n### Response:\n{a}" + eos

# 版本 B：新增 special token（进阶）
"<|user|>{q}<|assistant|>{a}<|end|>"
# 需要 resize embedding，新增 3 行随机初始化的 embedding
```

### 本阶段核心：loss mask

指令微调和「继续预训练」的**唯一本质区别**：

> **只在 assistant 的 token 上算 loss，user 部分全部设为 `-100` 忽略掉。**

必须写 `src/inspect_batch.py`，把 `input_ids` 和 `labels` 还原成人能读的形式打印出来，
肉眼确认 mask 边界正确（`-100` 的位置显示成 `[IGNORE]`）。

**验收标准**：随机抽 3 条样本，人能读懂对齐关系，且 user 部分全部被 mask。

---

## 阶段 4：手写 SFT 训练循环（1-2 天）

| 超参 | 建议值 | 说明 |
|---|---|---|
| 学习率 | **5e-5** | ⚠️ 千万别用预训练的 6e-4，那是灾难性遗忘的直通车 |
| 调度 | warmup 100 步 + cosine 衰减 | |
| 优化器 | AdamW，weight_decay 0.1 | |
| Batch | 16 序列 × 512 token，grad_accum 2 | 124M 全量微调在 7GB 显存下很宽裕 |
| 精度 | bf16 autocast | Blackwell 原生支持 |
| Epoch | 2-3 | 52k 条约 16 分钟/epoch |

**验收标准**：训练 loss 从 ~2.5 降到 ~1.0 附近（绝对数值不重要，趋势重要）。

---

## 阶段 5：评估与消融（1 天）

**定量**：留出 500 条不参与训练的样本，只算 assistant 部分的 loss。

**定性**：固定 20 条 prompt（含闲聊、事实问答、拒绝类），
base 模型 vs SFT 模型输出并排对比，人工打分。

### 三个消融实验

| 消融 | 对比 | 预期现象 |
|---|---|---|
| A | 有 loss mask vs 无 | 无 mask 的模型会**自己生成用户的问题**，变成两人对话模拟器 |
| B | 模板 A vs 模板 B | special token 更干净，但新增 embedding 需要更多步数收敛 |
| C | lr ∈ {2e-5, 5e-5, 2e-4} | lr 过高：忘掉通用能力；过低：学不会指令跟随 |

---

## 阶段 6：可选延伸

SFT 跑通后再决定是否做**偏好对齐**。

建议**跳过 RM + PPO**（8GB 显存要同时塞 policy / reward / ref / value 四个模型，
调试难度极高），直接上 **DPO**：

- 只需要 policy + reference 两个模型
- 同样的数据集能直接复用
- 能干净地分离出「指令微调」和「偏好对齐」各自的贡献

---

## 目录结构

```
gpt2/
├─ EXPERIMENT_PLAN.md     # 本文件
├─ README.md              # 实验日志，记录每次 run 的超参和结果
├─ requirements.txt
├─ .venv/                 # Python 3.13 虚拟环境
├─ src/
│  ├─ model.py            # 手写 GPT-2
│  ├─ load_weights.py     # HF -> 本地模型
│  ├─ verify_parity.py    # 与 HF 对拍（唯一用 transformers 的地方）
│  ├─ template.py         # 聊天模板 + loss mask
│  ├─ dataset.py
│  ├─ inspect_batch.py    # 人肉检查 mask
│  ├─ train_sft.py
│  ├─ generate.py
│  └─ eval.py
├─ data/
│  ├─ raw/                # alpaca.json 原始数据
│  └─ processed/          # train.jsonl / val.jsonl
├─ checkpoints/
└─ logs/
```

---

## 时间预算

约 **5-7 天**。其中阶段 1 的对拍和阶段 3 的 mask 检查吃掉了大半调试时间，
**别压缩这两步**。

---

## 进度追踪

- [x] 阶段 0：环境搭建
- [x] 阶段 1：手写 GPT-2 + 权重对拍
- [x] 阶段 2：~~证明「不会聊天」~~ → **改为单条数据的冒烟测试**（管线已验证跑通）
- [ ] 阶段 3：数据与模板
- [ ] 阶段 4：SFT 训练
- [ ] 阶段 5：评估与消融
- [ ] 阶段 6：可选延伸
