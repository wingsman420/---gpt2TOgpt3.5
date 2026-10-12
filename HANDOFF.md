# GPT-2 指令微调实验 — 交接文档

> **新会话开局读这一份，就能立刻接上进度。**
> 本文件是三份文档（规划 / 日志 / 交接）合并后的唯一记录。
> 最后更新：2026-10-12

---

## 一、项目目标

从零手写 GPT-2，加载官方预训练权重，然后自己实现**指令微调（SFT）**，
亲眼看到同一个模型在微调前后「不会聊天 → 会聊天」的行为差异。

参照文档：
- InstructGPT 论文 *Training language models to follow instructions with human feedback*
- karpathy/nanoGPT 与 "Let's reproduce GPT-2" 视频系列
- Alpaca 数据集与 Stanford Alpaca 的模板设计

---

## 二、已确定的选型（不要再重新讨论）

| 决策项 | 选择 | 说明 |
|---|---|---|
| 实现方式 | **全手写（nanoGPT 风格）** | 不依赖 transformers 定义模型 |
| 实验终点 | **分阶段：先跑通 SFT** | 跑通后再决定 RM+PPO 还是 DPO |
| 基座模型 | **英文 GPT-2 124M** | `openai-community/gpt2`，124,439,808 参数 |
| 分词器 | tiktoken（`gpt2` 编码） | 唯一不手写的组件 |
| 数据集 | `tatsu-lab/alpaca` | 约 52k 条英文指令数据 |
| 激活函数 | **`gelu_new`** | 实测确定，不是论文写的精确 `gelu` |

**为什么不用中文**：tiktoken 的 gpt2 词表是英文的，一个汉字要 2-3 个 byte 级
token，且模型没有中文先验。单条记忆能跑通，但做真实数据集效果会明显差于英文。

---

## 三、进度

- [x] **阶段 0：环境搭建** — 通过
- [x] **阶段 1：手写 GPT-2 + 权重对拍** — 通过（误差 7.629e-05）
- [x] **阶段 2：SFT 冒烟测试** — 通过（原计划"证明不会聊天"改为单条数据验证管线）
- [x] **阶段 3：准备正式数据集** — 通过（51,474 / 500，mask 全量校验 0 错误）
- [ ] **阶段 4：SFT 正式训练** ← **下一步**
- [ ] 阶段 5：评估与消融
- [ ] 阶段 6：可选延伸（DPO）

---

## 四、环境（本机约束）

| 项 | 值 |
|---|---|
| Python | 3.13.12（虚拟环境 `.venv`） |
| PyTorch | 2.11.0 + CUDA 12.8 |
| GPU | RTX 5060 Laptop / 8GB（Blackwell, **sm_120**） |
| 可用显存 | 约 6.9 GB |

### ⚠️ 三个环境坑

**1. PATH 里的 `python` 是 MSYS2 的 Python 3.14，没有 pip。**

必须显式用虚拟环境的解释器：

```bat
.venv\Scripts\python.exe script.py
```

**2. GPU 是 Blackwell（sm_120），必须用 CUDA 12.8+ 的 wheel。**

老版本 torch 不含 sm_120 的 kernel，装了也跑不了。
`src/check_env.py` 会验证这一点（并实跑一次矩阵乘法——
编译期支持 ≠ 运行期可用）。

**3. 访问 GitHub / HuggingFace 需要代理，端口 `127.0.0.1:7897`。**

git 没配代理，推送时要临时指定：

```bat
git -c http.proxy=http://127.0.0.1:7897 push origin main
```

HuggingFace 下载如果要加速，先设 `set HF_ENDPOINT=https://hf-mirror.com`。

### 重建环境

```bat
py -3.13 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe src\check_env.py
```

> torch 需要手动下载 wheel（网络原因），见 `requirements.txt` 顶部注释。

---

## 五、文件结构与分层规则

```
gpt2/
├─ HANDOFF.md              本文件（唯一的记录文档）
├─ requirements.txt
├─ .gitignore
├─ .venv/                  Python 3.13.12      （不入库，4.5G）
├─ weights/gpt2/           官方权重 522.7 MB   （不入库）
├─ sessions/               对话备份            （不入库，不可恢复）
├─ data/raw/smoke.jsonl    冒烟测试数据（1 条）
└─ src/                    ── 入口：只有这四个 ──
   ├─ gpt2.py              推理主入口 load() / complete()
   ├─ chat.py              对话 / 微调前后对比
   ├─ train_sft.py         SFT 训练循环
   ├─ check_env.py         环境验收
   ├─ _core/               ── 内部实现 ──
   │  ├─ __init__.py       分层规则说明
   │  ├─ model.py          手写 GPT-2
   │  └─ weights.py        下载 + 转置转换 + 装配
   ├─ _sft/                ── SFT 支持模块 ──
   │  ├─ __init__.py
   │  ├─ template.py       聊天模板 + loss mask（核心）
   │  └─ data.py           数据集 + batch（含自回归错位）
   └─ selfTry/             个人练习（独立，不参与管线）
```

### 分层规则

```
gpt2.py   ──▶  _core/           推理主入口，唯一的正规使用者
chat.py / train_sft.py ──▶ gpt2.py    应用代码只走主入口
                       ──▶ _sft/     SFT 支持模块
```

**应用代码不许直接 `import _core`。** 要拿模型就走 `from gpt2 import load`。

---

## 六、常用命令

```bat
cd /d "C:\Users\Administrator\Desktop\mind fragment\gpt2"

:: 环境验收
.venv\Scripts\python.exe src\check_env.py

:: 加载权重 + 自检（内部模块，一般不用直接跑）
.venv\Scripts\python.exe src\_core\weights.py --no-download

:: 用原始 GPT-2 续写
.venv\Scripts\python.exe src\gpt2.py "The capital of France is"

:: 看 loss mask 长什么样（训练前必查）
.venv\Scripts\python.exe src\_sft\template.py

:: 冒烟训练（单条数据，约 30 秒）
.venv\Scripts\python.exe src\train_sft.py --data data\raw\smoke.jsonl ^
    --steps 300 --lr 3e-4 --batch-size 8 --out checkpoints\smoke.pt

:: 微调前后对比
.venv\Scripts\python.exe src\chat.py --ckpt checkpoints\smoke.pt --compare --temperature 0.1

:: 数据准备（阶段 3，已跑过，一般不用重跑）
.venv\Scripts\python.exe src\_sft\prepare_alpaca.py

:: 看数据集统计 / 人肉检查 mask
.venv\Scripts\python.exe src\_sft\data.py data\processed\train.jsonl --show 0
.venv\Scripts\python.exe src\_sft\data.py data\processed\train.jsonl --show 3
```

---

## 七、阶段 1 记录 — 2026-09-15

手写 GPT-2 加载官方权重，与 HuggingFace 参考实现数值对拍。

```
activation = gelu_new   最大误差 7.629e-05   [PASS]
activation = gelu       最大误差 7.997e-02   [不同]
```

模型 124,439,808 参数（其中嵌入 39,383,808）。与官方实现**逐位一致**，
后续 SFT 出问题可以直接排除"模型本身写错了"。

### 误差量级诊断表（很有用，记下来）

| 误差 | 病因 |
|---|---|
| ~1e-5 | 正确 |
| ~1e-2 | 激活函数用错 |
| ~1e-1 | 转置或 bias 错 |
| ~1e+1 | 权重没加载上 / 形状被广播 |

### 三个发现

**1. Hub 上的 safetensors 没有 `transformer.` 前缀**

原以为 key 是 `transformer.wte.weight`，实际是 `wte.weight`——
因为 `openai-community/gpt2` 存的是 `GPT2Model` 的 state_dict
（`GPT2Model` 的属性就叫 `wte` / `wpe` / `h` / `ln_f`，没有 `transformer` 这一层）。

文件里另有 12 个 `h.N.attn.bias`（HF 的因果掩码 buffer，我们用
`is_causal=True` 现算，不需要）；`lm_head.weight` 不在文件里（与 `wte` 共享）。

**2. 形状不一致的减法会静默广播，不报错**

手写模型 `targets=None` 时只返回最后一个位置 `[B, 1, vocab]`，
HF 返回全部位置 `[B, T, vocab]`。相减被广播成「最后一个位置」对
「每一个位置」，误差从 1e-5 虚高到 **85.9**——看起来像模型全错了，
其实两个模型都对。

**教训：对拍前先 assert 形状一致。**

**3. 激活函数以实测为准**

GPT-2 论文写的是精确 GELU，但 HF 的实现用 `gelu_new`（tanh 近似）。
按论文记忆去写会得到 8e-2 的误差，然后你会去怀疑转置——而转置根本没错。

### 权重加载的两个坑（`_core/weights.py`）

- HF 用 **Conv1D**，权重是 `[in, out]`；我们用 `nn.Linear`，要 `[out, in]` → **必须转置**
- **`c_proj` 是方阵**（768×768），转置前后形状都合法，`load_state_dict` **不报错**。
  12 层 × 2 = 24 个方阵都可能悄悄错，只有数值对拍能发现

---

## 八、阶段 2 记录 — 2026-09-17

**任务调整**：原计划"证明它不会聊天并写报告"，改为**一条数据的冒烟测试**——
先用最小数据验证整条训练管线能跑通。

数据 `data/raw/smoke.jsonl`，仅一条：

```json
{"instruction": "Who are you?", "response": "I am ChatGPT."}
```

结果：loss 0.9128 → 0.0000，**300 步 27 秒**。

| 模型 | 对 `Who are you?` 的回答 |
|---|---|
| 原始 GPT-2 | `### What is your name?\n\n### What is your age?\n\n### What is` |
| 微调后 | **`I am ChatGPT.`** |

原始模型的输出**精确印证了实验预期**：它不回答，而是继续编造同格式的问题——
因为它把这个模板当成了另一段要补全的网页文本。

### 抓到关键 bug：训练目标错位一位

第一版训练 loss 同样归零，但生成出来是 `........`。

**原因**：`logits[i]` 预测的是**第 i+1 个** token（自回归的定义），
但 `labels` 被原样传进 loss，`labels[i]` 对应的是第 i 个。**差了一位。**

结果模型学会了"复制当前 token"——一个平凡任务，loss 迅速归零，
但生成完全是垃圾。

**修复**：在 `_sft/data.py` 的 `make_batch` 里错位，`x = ids[:-1]`，`y = labels[1:]`。

> **教训**：这类 bug 最阴险——**训练指标完美、产出全错**。
> 如果跳过冒烟测试直接上 52k 数据，会浪费几小时才发现。
> 这就是"先跑通管线再加数据"的价值。

### 另一个坑：tiktoken 拒绝编码特殊 token

`enc.encode("<|endoftext|>")` 会抛 `ValueError`——tiktoken 默认**拒绝**把
特殊 token 当普通文本编码，防止你无意中把它混进训练数据。
我们是故意要用，所以显式放行：`enc.encode(text, allowed_special={EOS_TEXT})`。

---

## 九、阶段 3 记录 — 2026-10-12

把 Alpaca 52k 转成我们的 SFT jsonl 格式。

### 实测数据画像

```
tatsu-lab/alpaca          52,002 条，HF 上存成 parquet（不是 json）
  字段                    instruction / input / output / text
  input 非空              20,679 条 (39.8%)
  output 为空             28 条

转换后                    51,974 条（丢 28 条空 output）
  合并了 input 的          21,013 条 (40.4%)
  train / val             51,474 / 500（seed=42，可复现）

token 长度（抽样 5,000）   中位 72，p90 156，p99 303，max 686
  max_len=512 只丢 0.1%    → 512 是合适的选择
参与 loss 的 token         67.0%（健康比例）
```

### 三个实现要点

**1. `input` 字段必须合并进 instruction**

约四成样本的 `input` 非空，装的是任务素材。不合并的话指令就不完整——
比如「把下面这句话分类」却没有「下面这句话」，模型只能学到
"看到 classify 就输出 positive"。

合并方式（`_sft/prepare_alpaca.py` 的 `merge_input`）：

```
### Instruction:
{instruction}

{input}

### Response:
{output}
```

**2. 不用 Alpaca 自己的 `text` 字段**

`text` 是 Alpaca 拼好的模板字符串（带一段 "Below is an instruction that
describes a task..." 前言），和我们的不一样。
**训练模板必须和推理模板逐字节一致**，所以模板一律由 `template.py` 生成。

**3. 数据集现在是 parquet，需要 `pyarrow`**

原来 `alpaca_data.json` 的路径已经 404。HF 上只有
`data/train-00000-of-00001-....parquet`。`pyarrow` 已加进环境。

### 验收（这一步不能跳）

```bat
:: 看统计
.venv\Scripts\python.exe src\_sft\data.py data\processed\train.jsonl --show 0

:: 人肉看 mask
.venv\Scripts\python.exe src\_sft\data.py data\processed\train.jsonl --show 3
```

实测结果：

| 检查 | 结果 |
|---|---|
| 合并 input 的样本，边界是否精确 | ✅ 位置 28 `\n` 是 `[IGNORE]`，29 开始 ✓ |
| 提问部分是否全部 mask | ✅ 29/29 |
| train / val 是否重叠 | ✅ 0 条 |
| **全量扫描 51,474 条的 mask 边界** | ✅ **0 条错误** |

最后一项是全量校验，不是抽样。

---

## 十、剩余阶段规划

### 阶段 4：SFT 正式训练 ← **下一步**

模板和 loss mask 已在阶段 2 做完（`src/_sft/template.py`），
所以只剩一件事：**把 Alpaca 52k 转成 `{"instruction":..., "response":...}` 的 jsonl**。

**关键决策：`input` 字段**

Alpaca 有四个字段：`instruction` / `input` / `output` / `text`。
约 **四成样本的 `input` 非空**，装的是任务需要的额外素材。

例如 `instruction="Classify the sentence"`、`input="The movie was great"`、
`output="positive"`。**忽略 input 的话，要分类的那句话就不见了**，
模型只能学到"看到 classify 就输出 positive"。

所以 `input` 必须合并进 instruction。

**产出**：`data/processed/train.jsonl`（约 51.5k）+ `val.jsonl`（500 条）

**验收**（两步都不能跳）：

```bat
:: 看统计
.venv\Scripts\python.exe src\_sft\data.py data\processed\train.jsonl --show 0

:: 人肉看 mask，确认提问部分全是 [IGNORE]
.venv\Scripts\python.exe src\_sft\data.py data\processed\train.jsonl --show 3
```

### 阶段 4：SFT 正式训练

```bat
.venv\Scripts\python.exe src\train_sft.py --data data\processed\train.jsonl ^
    --epochs 3 --lr 5e-5 --out checkpoints\sft.pt
```

| 超参 | 值 | 说明 |
|---|---|---|
| 学习率 | **5e-5** | ⚠️ 千万别用预训练的 6e-4，那是灾难性遗忘的直通车 |
| 调度 | warmup + cosine | |
| 优化器 | AdamW，weight_decay 0.1 | 权重加衰减，bias/LayerNorm 不加 |
| Batch | 8 × 512 token | 7GB 显存下很宽裕 |
| 精度 | bf16 autocast | Blackwell 原生支持 |

预计 **1.5~3 小时**（19,000 步）。

### 阶段 5：评估与消融

- **定量**：val 集上只算 assistant 部分的 loss，看有没有过拟合
- **定性**：固定 20 条 prompt，base vs SFT 并排对比

三个消融：

| 消融 | 对比 | 预期现象 |
|---|---|---|
| A | 有 loss mask vs 无 | 无 mask 的模型会**自己生成用户的问题**，变成两人对话模拟器 |
| B | 纯文本模板 vs special token | 后者更干净，但要 resize embedding |
| C | lr ∈ {2e-5, 5e-5, 2e-4} | 过高忘掉通用能力；过低学不会指令跟随 |

### 阶段 6：可选延伸

建议**跳过 RM + PPO**（8GB 显存要同时塞 policy/reward/ref/value 四个模型），
直接上 **DPO**——只需 policy + reference 两个模型，数据集能直接复用。

---

## 十一、Git 工作流

仓库：https://github.com/wingsman420/---gpt2TOgpt3.5

```
main      最新进度
stage-2-sft       SFT 冒烟测试完成
stage-1-model     手写 GPT-2 + 权重对拍完成
stage-0-env       环境搭建完成
```

**累积式**：切到 `stage-N` 就等于回到那个阶段的目录状态。

```bat
git branch -v
git checkout stage-1-model
git checkout main
git log --oneline --all --graph
```

**为什么是累积而不是互相独立**：阶段之间有 import 依赖（`train_sft.py`
要 `import model`）。如果阶段 2 分支里没有阶段 1 的文件，代码根本跑不起来。

**推送要带代理**：

```bat
git -c http.proxy=http://127.0.0.1:7897 push origin main
```

**切分支前确保 `git status` 干净**，否则未提交的改动会跟着跑到另一个分支。

---

## 十二、下一步

**阶段 4：SFT 正式训练**

数据已就绪（`data/processed/train.jsonl`，51,474 条），直接跑：

```bat
.venv\Scripts\python.exe src\train_sft.py --data data\processed\train.jsonl ^
    --epochs 3 --lr 5e-5 --out checkpoints\sft.pt
```

**注意事项：**

- ⚠️ **lr 必须是 5e-5**，不是冒烟测试的 3e-4。微调用大学习率会灾难性遗忘
- 预计 **1.5~3 小时**（约 19,000 步）
- 训完用 val 集评估：`src/_sft/data.py data/processed/val.jsonl` 看 loss
- 定性对比：`src/chat.py --ckpt checkpoints/sft.pt --compare`

**训练时要盯的：**

| 现象 | 含义 |
|---|---|
| loss 平稳下降 | 正常 |
| loss 突然变成 nan | 学习率太大或数据有问题，先看梯度裁剪 |
| loss 降到很低但生成是垃圾 | 目标错位（阶段 2 踩过，见 `_sft/data.py` 的错位注释） |

---

## 附：一个值得记住的通用教训

在做完阶段 2 后做过一次分层重构，过程中抓到三个真 bug：

1. **移动文件后 `PROJECT_ROOT` 层级算错** —— 相对路径的层级会跟着文件一起变
2. **检查点里 pickle 了自定义类对象** —— pickle 会记下类的模块路径，
   源文件一改名，**所有历史检查点永久失效**（报 `ModuleNotFoundError`，
   看起来像安装问题，很容易查错方向）。
   改法：存 `asdict(model.config)` 而不是对象本身。附带好处是可以开
   `torch.load(..., weights_only=True)`，更安全
3. **路径散落各处** —— `sys.path` 插入、错误提示、文档里的命令都是硬编码字符串

> **通用教训：不要把自定义类的对象塞进检查点。存值，不存对象。**
