# 交接文档

> 新会话开局读这一份，就能立刻接上进度，不用重新翻聊天记录。
>
> - 主会话 ID：`441ac2bf-00e3-4a9f-bc34-04437dea5616`
> - 备份时间：2026-09-16
> - 完整对话稿：`sessions/chat-441ac2bf-*.md`
> - 原始记录：`sessions/raw-441ac2bf-20260916.jsonl`

---

## 一、项目目标

从零手写 GPT-2，加载官方预训练权重，然后自己实现**指令微调（SFT）**，
亲眼看到同一个模型在微调前后「不会聊天 → 会聊天」的行为差异。

完整规划见 **`EXPERIMENT_PLAN.md`**。

## 二、已确定的选型（不要再重新讨论）

| 决策项 | 选择 |
|---|---|
| 实现方式 | 全手写（nanoGPT 风格），不依赖 transformers 定义模型 |
| 实验终点 | 分阶段：先跑通 SFT，跑通后再决定 RM+PPO 或 DPO |
| 基座模型 | 英文 GPT-2 124M（`openai-community/gpt2`） |
| 分词器 | tiktoken 的 `gpt2` 编码（唯一不手写的组件） |
| 数据集 | `tatsu-lab/alpaca`（英文，52k 条） |
| 激活函数 | **gelu_new**（实测确定，不用精确 gelu） |

## 三、进度

- [x] **阶段 0：环境搭建** — 通过
- [x] **阶段 1：手写 GPT-2 + 权重对拍** — 通过
- [x] **阶段 2：冒烟测试** — 通过（原计划"证明不会聊天"改为单条数据验证管线）
- [ ] 阶段 3：数据与模板 ← **下一步**（模板和 mask 已在阶段 2 顺带做完）
- [ ] 阶段 4：SFT 训练
- [ ] 阶段 5：评估与消融
- [ ] 阶段 6：可选延伸

## 四、阶段 1 的关键结论

```
activation = gelu_new   最大误差 7.629e-05   [PASS]
activation = gelu       最大误差 7.997e-02   [失败]
```

模型 124,439,808 参数。与 HuggingFace 参考实现逐位一致，**可以完全信任**。
后续 SFT 出问题，直接去查数据/掩码/超参，不必怀疑模型本身。

### 误差量级诊断表

| 误差 | 病因 |
|---|---|
| ~1e-5 | 正确 |
| ~1e-2 | 激活函数用错 |
| ~1e-1 | 转置或 bias 错 |
| ~1e+1 | 权重没加载上 / 形状被广播 |

## 五、已经踩过的坑（别重踩）

1. **Hub 上的 safetensors 没有 `transformer.` 前缀**。存的是 `GPT2Model` 的
   state_dict，key 是 `wte.weight` / `h.0.attn...` / `ln_f.weight`。
   还有 12 个 `h.N.attn.bias`（因果掩码常量，用不上），
   以及没有 `lm_head.weight`（与 `wte` 共享）。

2. **形状不一致的减法会静默广播，不报错**。`[B,1,V] - [B,T,V]` 会把误差
   从 1e-5 虚高到 85.9。对拍前必须 `assert` 形状一致。

3. **`c_proj` 是方阵**，转置前后形状都合法，`load_state_dict` 不报错。
   12 层 × 2 = 24 个方阵，每个都可能悄悄错。只有数值对拍能发现。

4. **本机 `python` 命令是 MSYS2 的 Python 3.14，没有 pip**。
   必须用 `.venv/Scripts/python.exe`。

## 六、文件清单

```
gpt2/
├─ EXPERIMENT_PLAN.md      六阶段完整规划
├─ HANDOFF.md              本文件
├─ README.md               实验日志（含阶段 1、2 记录）
├─ requirements.txt
├─ .gitignore
├─ .venv/                  Python 3.13.12         （不入库）
├─ weights/gpt2/           官方权重 522.7 MB      （不入库）
├─ checkpoints/            训练产物               （不入库）
├─ sessions/               对话备份               （不入库）
├─ data/raw/smoke.jsonl    冒烟测试数据（1 条）
├─ src/                    ── 入口：给人看的只有这四个 ──
│  ├─ gpt2.py              推理主入口 load() / complete()
│  ├─ chat.py              对话 / 微调前后对比
│  ├─ train_sft.py         SFT 训练循环
│  ├─ check_env.py         环境验收
│  ├─ _core/               ── 内部实现 ──
│  │  ├─ __init__.py       分层规则说明
│  │  ├─ model.py          手写 GPT-2
│  │  └─ weights.py        下载 + 转置转换 + 装配
│  ├─ _sft/                ── SFT 支持模块 ──
│  │  ├─ __init__.py
│  │  ├─ template.py       聊天模板 + loss mask（核心）
│  │  └─ data.py           数据集 + batch（含自回归错位）
│  └─ selfTry/             个人练习（独立，不参与管线）
├─ tests/
│  └─ test_parity.py       与 HF 数值对拍
└─ learning/               教学脚本
   ├─ walkthrough.py       推理全流程走读
   └─ sampling_demo.py     temperature / top_k 的作用
```

### 分层规则

```
gpt2.py  ──▶  _core/          推理主入口，唯一的正规使用者
tests/   ──▶  _core/          测试豁免：对拍需要 convert_state_dict
learning/──▶  _core/          教学豁免：走读需要逐层打印张量
chat.py / train_sft.py ──▶ gpt2.py    应用代码只走主入口
                       ──▶ _sft/       SFT 支持模块（下划线=非入口，非禁止）
```

**应用代码不许直接 `import _core`。** 要拿模型就走 `from gpt2 import load`。

> `attention_bridge.py` 和 `export_chat.py` 曾在 `src/` 下，
> 于 2026-09-17 从磁盘上消失（非本会话删除），未还原。

## 七、常用命令

```bat
cd /d "C:\Users\Administrator\Desktop\mind fragment\gpt2"

:: 环境验收
.venv\Scripts\python.exe src\check_env.py

:: 加载权重 + 自检（内部模块，一般不用直接跑）
.venv\Scripts\python.exe src\_core\weights.py --no-download

:: 数值对拍
.venv\Scripts\python.exe tests\test_parity.py

:: 用原始 GPT-2 续写
.venv\Scripts\python.exe src\gpt2.py "The capital of France is"

:: 看 loss mask 长什么样（训练前必查）
.venv\Scripts\python.exe src\_sft\template.py

:: 冒烟训练（单条数据，约 30 秒）
.venv\Scripts\python.exe src\train_sft.py --data data\raw\smoke.jsonl ^
    --steps 300 --lr 3e-4 --batch-size 8 --out checkpoints\smoke.pt

:: 微调前后对比
.venv\Scripts\python.exe src\chat.py --ckpt checkpoints\smoke.pt --compare --temperature 0.1

:: 教学脚本（在 learning/ 下）
.venv\Scripts\python.exe learning\walkthrough.py
.venv\Scripts\python.exe learning\sampling_demo.py
```

## 八、下一步要做什么

**阶段 3：准备正式数据集**

模板和 loss mask 已经在阶段 2 做完了（`src/template.py`）。
阶段 3 只剩一件事：**把 Alpaca 52k 转成同样的 jsonl 格式**。

```json
{"instruction": "...", "response": "..."}
```

产出 `data/processed/train.jsonl`（和 `val.jsonl`，留 500 条做评估），
然后直接把 `--data` 指过去就能跑正式训练：

```bat
.venv\Scripts\python.exe src\train_sft.py --data data\processed\train.jsonl ^
    --epochs 3 --lr 5e-5 --out checkpoints\sft.pt
```

⚠️ 注意正式训练的学习率是 **5e-5**，不是冒烟测试的 3e-4。
微调用大学习率会灾难性遗忘。

## 九、Git 工作流

每个阶段一个分支，**累积式**：切到 `stage-N` 就等于回到那个阶段的目录状态。

```
main           最新进度（当前 = stage 2）
stage-2-sft    SFT 冒烟测试完成
stage-1-model  手写 GPT-2 + 权重对拍完成
stage-0-env    环境搭建完成
```

```bat
git branch -v               :: 看有哪些分支
git checkout stage-1-model  :: 回到阶段 1（src/ 里只剩 5 个文件，但依然能跑）
git checkout main           :: 回到最新
git log --oneline --all --graph
```

**为什么是累积而不是互相独立**：阶段之间有 import 依赖 ——
`train_sft.py` 要 `import model`。如果阶段 2 分支里没有阶段 1 的文件，
代码根本跑不起来。所以每个分支都包含它之前所有阶段的内容。

**不要 `git checkout` 的同时改文件**：切分支前确保 `git status` 是干净的，
否则改动会跟着你跑到另一个分支上。

**入库策略**：权重（522MB）、`.venv`、检查点、日志、对话备份都不入库，
见 `.gitignore`。`data/raw/smoke.jsonl` 入库，`data/processed/` 不入库。
