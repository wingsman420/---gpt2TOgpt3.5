"""GPT-2 的内部实现：手写网络结构 + 权重装配。

⚠️ 分层规则（重要）
===================

**面向使用者的代码，不要直接 import 这个包里的东西。**

要拿模型，走推理主入口：

    from gpt2 import load, complete

唯一允许深入本包的只有三类：

    gpt2.py       推理主入口 —— 本包的正规使用者
    tests/        测试代码 —— 需要检查内部细节
    learning/     教学代码 —— 需要展示内部细节

比如 `tests/test_parity.py` 要拿到 `convert_state_dict` 才能做数值对拍，
`learning/walkthrough.py` 要拿到 `GPT` 类才能逐层打印张量形状。
不给看它们就干不了活。

所以规则准确的表述是：**"对外只暴露推理入口"里的"外"，指的是使用者
和应用代码；测试与教学有豁免权。** 这是常见的分层做法，不是破例。


模块
====

    model.py     手写 GPT-2 网络结构（12 层 Transformer）
    weights.py   下载官方权重 → 转置转换 → 装配进模型
"""
