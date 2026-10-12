#这是个人尝试的文件，首先我们先梳理你要写什么
### 1，加载分词器，enc = tiktoken.get_encoding("gpt2")
### 2，加载模型，model 
### 3，
import argparse
import torch
from model import LLM




def main():
    print ("正在运行模型...")
    parser = argparse.ArgumentParser()
    parser.add_argument("--file",nargs="?", default=None,const="input.txt",help="输入文本")
    args = parser.parse_args()

    if args.file:
        print(f"从文件 {args.file} 读取输入文本")
        with open(args.file,"r",encoding="utf-8") as f:
            input_text = f.read()
    else:
        input_text = input("请输入文本：")

    GPT2 = LLM(1.0,50)#这里改温度和topk
    print("checkpoint1")
    GPT2.load_model(LLM.MODEL_PATH, device= "cuda" if torch.cuda.is_available() else "cpu")
    print("checkpoint2")
    GPT2.load_prompt(input_text)
    for _ in range(50):#这里改生成的长度
        next_token = GPT2.generate()
        if not next_token:
            break
    print(GPT2.getprompt())#输出最终的文本

if __name__ == "__main__":
    main()