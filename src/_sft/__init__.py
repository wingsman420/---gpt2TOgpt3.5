"""指令微调（SFT）的支持模块。

    template.py   聊天模板 + loss mask —— 整个实验的核心
    data.py       数据集加载 + batch 构造（含自回归错位）

使用方是 `train_sft.py`。

关于 `_` 前缀
=============

这里的 `_` 不是"禁止访问"（和 `_core` 不同），而是**"这些不是入口，
是支持文件"**。

`src/` 下直接可见的四个文件是给人看的入口：

    gpt2.py        推理主入口
    chat.py        对话
    train_sft.py   训练
    check_env.py   环境验收

其余实现细节沉在子包里，让入口一眼可见。
"""
