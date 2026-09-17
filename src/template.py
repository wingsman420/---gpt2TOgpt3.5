"""聊天模板 + loss mask。

这是整个实验**最核心的一个文件**。

指令微调和「继续预训练」的唯一本质区别就在 loss mask：

    继续预训练：对**每一个** token 算 loss
    指令微调：  只对 **assistant 回复部分**算 loss，用户的提问全部忽略

不 mask 的话，模型会连用户的提问一起学 —— 结果就是它自己生成用户的问题，
变成一个人分饰两角的对话模拟器。阶段 5 的消融实验 A 会把这个现象做出来。

模板格式（Alpaca 风格纯文本，不新增 special token）
====================================================

    ### Instruction:
    Who are you?

    ### Response:
    I am ChatGPT.<|endoftext|>

为什么先做这个版本：不需要 resize embedding，风险最低。
进阶版（新增 <|user|> / <|assistant|> token）留到后面再对比。
"""

from __future__ import annotations

# 用 GPT-2 自带的特殊 token 当分隔符和终止符，避免新增词表
EOS_TEXT = "<|endoftext|>"

PROMPT_TEMPLATE = "### Instruction:\n{instruction}\n\n### Response:\n"


def build_prompt(instruction: str) -> str:
    """构造提问部分的文本。**推理时也必须用这个函数**，逐字节一致。"""
    return PROMPT_TEMPLATE.format(instruction=instruction)


def build_example(enc, instruction: str, response: str) -> tuple[list[int], list[int]]:
    """把一条对话转成 (input_ids, labels)。

    labels 里等于 -100 的位置不参与 loss 计算（PyTorch 的 ignore_index）。

    返回：
        input_ids  模型看到的全部 token
        labels     与 input_ids 等长，提问部分是 -100，回复部分是真实 id
    """
    prompt = build_prompt(instruction)
    full_text = prompt + response + EOS_TEXT

    # tiktoken 默认**拒绝**把特殊 token 当普通文本编码，抛 ValueError。
    # 这是防止你无意中把 <|endoftext|> 混进训练数据的设计。
    # 我们是故意要用它，所以显式放行这一个。
    allow = {EOS_TEXT}

    prompt_ids = enc.encode(prompt)
    full_ids = enc.encode(full_text, allowed_special=allow)

    # 分词边界校验。
    # BPE 理论上可能把边界处的字符合并成一个 token，导致
    # 「先编码 prompt 再拼接」和「整体编码」结果不一致。
    # 这里断言两者一致；不一致就必须换做法，不能装作没看见。
    n_prompt = len(prompt_ids)
    if full_ids[:n_prompt] != prompt_ids:
        raise ValueError(
            "分词边界对不上：prompt 和整体编码的前缀不一致。\n"
            f"  prompt     : {prompt_ids}\n"
            f"  full[:n]   : {full_ids[:n_prompt]}\n"
            "需要改用「分别编码再拼接」的构造方式。"
        )

    labels = [-100] * n_prompt + full_ids[n_prompt:]
    return full_ids, labels


def inspect(enc, instruction: str, response: str) -> None:
    """人肉检查 mask 是否正确。

    阶段 3 的验收要求：随机抽样本，人能读懂对齐关系，
    且提问部分全部被 mask。
    """
    ids, labels = build_example(enc, instruction, response)

    print(f"\n{'─' * 74}")
    print(f"instruction: {instruction!r}")
    print(f"response   : {response!r}")
    print(f"{'─' * 74}")
    print(f"{'位置':<6}{'token id':<12}{'参与loss':<12}{'文本'}")
    print(f"{'─' * 74}")

    for i, (tid, lab) in enumerate(zip(ids, labels)):
        text = enc.decode([tid]).replace("\n", "\\n")
        if lab == -100:
            print(f"{i:<6}{tid:<12}{'[IGNORE]':<12}{text!r}")
        else:
            print(f"{i:<6}{tid:<12}{'✓':<12}{text!r}")

    n_train = sum(1 for x in labels if x != -100)
    print(f"{'─' * 74}")
    print(f"总长 {len(ids)} token，其中 {n_train} 个参与 loss，{len(ids) - n_train} 个被 mask")
    print(f"\n⚠️ 检查点：'### Instruction:' 到 '### Response:\\n' 之间的所有 token")
    print(f"   必须全部是 [IGNORE]。如果混进了 ✓，模型就会去学怎么生成提问。")


if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import tiktoken

    enc = tiktoken.get_encoding("gpt2")
    inspect(enc, "Who are you?", "I am ChatGPT.")
