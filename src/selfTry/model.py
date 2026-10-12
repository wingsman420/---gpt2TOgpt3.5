import os
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForCausalLM, GPT2Tokenizer

# 代理要在任何联网请求之前设置好（HuggingFace 下载要走它）
os.environ["HTTP_PROXY"] = "http://127.0.0.1:7897"
os.environ["HTTPS_PROXY"] = "http://127.0.0.1:7897"

class LLM:
    block_size: int = 1024  # 最大上下文长度，等于位置嵌入的行数
    vocab_size: int = 50257  # tiktoken 的 gpt2 词表大小
    tokenizer = None
    model = None
    config = None
    tokened_prompt = None
    temperature: float = 1.0
    top_k: int = 50

    # ---------------------------------------------------------------
    # 设备。
    #
    # 改进点：原来这个属性只在 load_model() 里被创建（self.device = device），
    # 意味着只要在 load_model() 之前调用 load_prompt() 或 generate()，
    # 就会报 AttributeError: 'LLM' object has no attribute 'device'。
    #
    # 这是「隐式依赖」—— 一个「必须先调 A 才能调 B」的规则，只存在于你脑子里，
    # 代码里看不出来，忘了就炸。给一个类级别的默认值，属性就**永远存在**了，
    # load_model() 会用真实设备把它覆盖掉。
    #
    # 为什么放在类属性而不是 __init__ 里：本类所有状态（tokenizer / model /
    # config / tokened_prompt ...）都是类属性风格，保持一致。
    #
    # 补充：实例属性会遮蔽同名的类属性，所以 self.device = "cuda" 之后，
    # 类上的 "cpu" 对**这个实例**就不再生效了，也不会污染别的实例。
    # ---------------------------------------------------------------
    device: str = "cpu"

    MODEL_PATH = "C:\\Users\\Administrator\\Desktop\\mind fragment\\gpt2\\weights\\gpt2"

    def __init__(self,temperature: float = 1.0,top_k: int = 50):
        print("正在加载分词器...")
        self.tokenizer = GPT2Tokenizer.from_pretrained(self.MODEL_PATH)
        print("加载分词器完成")
        self.temperature = temperature
        self.top_k = top_k


    def generate(self) -> str:
        with torch.no_grad():
            logits = self.model(self.tokened_prompt).logits

        # ---------------------------------------------------------------
        # 原始写法（保留作对照）
        # ---------------------------------------------------------------
        # indices_to_remove = logits < torch.topk(logits, self.top_k)[0][..., -1, None]
        # logits = logits.masked_fill(indices_to_remove, -float('Inf'))
        # probs = F.softmax(logits[0, -1] / self.temperature, dim=-1)
        # ---------------------------------------------------------------
        #
        # 为什么改：
        #
        #   logits 的形状是 [1, T, 50257]。
        #   torch.topk(logits, k) 默认沿着**最后一维**操作，而且会遍历
        #   **每一个位置**，得到 [1, T, k]。但我们后面只用 logits[0, -1]
        #   这一个位置 —— 前面 T-1 个位置算的全是白工。
        #
        #   原写法里 [..., -1, None] 是在绕弯：
        #       [..., -1]   取最后一维的第 k 个值   → [1, T]
        #       [None]      在最前面插一个维度       → [1, T, 1]
        #   之所以要插这个维度，是为了能和 [1, T, 50257] 广播相减
        #   （形状不齐的话 PyTorch 会按最后一维对齐，结果就错了）。
        #   绕这一大圈，只是为了拿到一个位置的阈值。
        #
        #   两种写法**数值上完全等价** —— 同一个位置、同一个门槛、同样的
        #   softmax 输入。差别只在算力：序列越长，白算的部分越多。
        # ---------------------------------------------------------------

        # 新写法：先把最后一个位置取出来，只在这一个位置上做 top-k
        last_logits = logits[0, -1]                                     # [50257]
        threshold = torch.topk(last_logits, self.top_k)[0][-1]          # 第 k 大的值 = 门槛
        last_logits = last_logits.masked_fill(last_logits < threshold, -float("Inf"))
        probs = F.softmax(last_logits / self.temperature, dim=-1)
        next_token_id = torch.multinomial(probs, 1).item()
        next_token_tensor = torch.tensor([[next_token_id]],device=self.device)
        self.tokened_prompt = torch.cat((self.tokened_prompt, next_token_tensor), dim=1)
        if next_token_id == self.tokenizer.eos_token_id:
            return ""
        next_token_text = self.tokenizer.decode([next_token_id])
        return next_token_text

        

    def load_model(self, local_dir: Path, device: str):
        """加载模型"""
        self.device = device                                    # ① 存成实例属性，覆盖类上那个默认的 "cpu"
        self.config = AutoConfig.from_pretrained(local_dir)     # ② 用参数，不用 self.MODEL_PATH
        self.model = AutoModelForCausalLM.from_pretrained(local_dir)
        self.model.to(device)                                   # ③ 搬到目标设备
        self.model.eval()                                       # ④ 切推理模式
        return self.model

    def load_prompt(self,prompt:str)->any:
        "加载提示词"
        self.tokened_prompt = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        return self.tokened_prompt

    def getprompt(self)->str:
        "获取提示词"
        return self.tokenizer.decode(self.tokened_prompt[0])
