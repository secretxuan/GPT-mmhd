"""
=================================================================================
文本生成脚本
=================================================================================

这个脚本演示如何使用训练好的 GPT 模型生成文本。

使用方式：
    python generate.py --checkpoint checkpoints/model.pt --prompt "林黛玉" --max_tokens 100

参数说明：
    --checkpoint: 模型检查点路径
    --prompt: 起始文本
    --max_tokens: 生成的最大 token 数
    --temperature: 采样温度（越高越随机）
    --top_k: top-k 采样参数
"""

import torch
import sys
import os

# 添加父目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import GPT, GPTConfig


class SimpleTokenizer:
    """简单的字符级分词器"""

    def __init__(self, text=None):
        if text is not None:
            self.chars = sorted(list(set(text)))
            self.char_to_id = {ch: i for i, ch in enumerate(self.chars)}
            self.id_to_char = {i: ch for i, ch in enumerate(self.chars)}
            self.vocab_size = len(self.chars)

    def encode(self, text):
        return [self.char_to_id[ch] for ch in text]

    def decode(self, ids):
        return ''.join([self.id_to_char[i] for i in ids])

    @classmethod
    def load(cls, path):
        import pickle
        with open(path, 'rb') as f:
            data = pickle.load(f)
        tokenizer = cls()
        tokenizer.chars = data['chars']
        tokenizer.char_to_id = data['char_to_id']
        tokenizer.id_to_char = data['id_to_char']
        tokenizer.vocab_size = data['vocab_size']
        return tokenizer

    def save(self, path):
        import pickle
        data = {
            'chars': self.chars,
            'char_to_id': self.char_to_id,
            'id_to_char': self.id_to_char,
            'vocab_size': self.vocab_size
        }
        with open(path, 'wb') as f:
            pickle.dump(data, f)


def generate_text(model, tokenizer, prompt, max_tokens=100, temperature=0.8, top_k=40):
    """
    生成文本

    Args:
        model: GPT 模型
        tokenizer: 分词器
        prompt: 起始文本
        max_tokens: 生成的最大 token 数
        temperature: 采样温度
        top_k: top-k 采样

    Returns:
        生成的文本（包含 prompt）
    """
    model.eval()

    # 编码 prompt
    idx = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long)

    # 生成
    with torch.no_grad():
        output = model.generate(
            idx,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k
        )

    # 解码
    return tokenizer.decode(output[0].tolist())


def demo_untrained_generation():
    """演示未训练模型的生成（随机输出）"""
    print("=" * 60)
    print("【1. 未训练模型的生成（随机）】")
    print("=" * 60)

    # 读取数据创建分词器
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'hongloumeng.txt')
    with open(data_path, 'r', encoding='utf-8') as f:
        text = f.read()

    tokenizer = SimpleTokenizer(text)
    print(f"词汇表大小: {tokenizer.vocab_size}")

    # 创建模型
    config = GPTConfig(
        vocab_size=tokenizer.vocab_size,
        block_size=256,
        n_layer=4,
        n_head=4,
        n_embd=128,
        dropout=0.0  # 生成时不使用 dropout
    )
    model = GPT(config)

    # 生成文本
    prompt = "林黛玉"
    print(f"\nPrompt: {prompt}")
    print("生成中...")

    output = generate_text(model, tokenizer, prompt, max_tokens=50, temperature=0.8)

    print(f"\n生成结果:")
    print(output)

    print("""
注意：这是随机初始化的模型，输出是乱码。
需要训练后才能生成有意义的文本！
    """)


def demo_sampling_strategies():
    """演示不同的采样策略"""
    print("\n" + "=" * 60)
    print("【2. 采样策略】")
    print("=" * 60)

    print("""
生成文本时，有多种采样策略：

1. 贪心解码 (Greedy)
   - 每次选择概率最高的 token
   - 确定性强，但输出可能重复、无聊

2. 温度采样 (Temperature)
   - logits = logits / temperature
   - temperature 越低，越确定（接近贪心）
   - temperature 越高，越随机（更有创意）
   - 推荐：0.7 ~ 1.0

3. Top-k 采样
   - 只从概率最高的 k 个 token 中采样
   - 过滤掉不太可能的 token
   - 推荐：k = 40

4. Top-p (Nucleus) 采样
   - 从累积概率达到 p 的最小 token 集合中采样
   - 比 top-k 更灵活
   - 推荐：p = 0.9

实际使用：温度采样 + Top-k 组合效果最好
    """)


def demo_generation_params():
    """演示生成参数的影响"""
    print("\n" + "=" * 60)
    print("【3. 生成参数对比】")
    print("=" * 60)

    # 读取数据
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'hongloumeng.txt')
    with open(data_path, 'r', encoding='utf-8') as f:
        text = f.read()

    tokenizer = SimpleTokenizer(text)

    config = GPTConfig(
        vocab_size=tokenizer.vocab_size,
        block_size=256,
        n_layer=2,
        n_head=2,
        n_embd=64,
        dropout=0.0
    )
    model = GPT(config)

    prompt = "贾宝玉"

    # 不同的温度
    temperatures = [0.5, 0.8, 1.2]
    print(f"Prompt: '{prompt}'\n")

    for temp in temperatures:
        output = generate_text(model, tokenizer, prompt, max_tokens=30,
                               temperature=temp, top_k=None)
        print(f"Temperature {temp}:")
        print(f"  {output}\n")


def explain_generation_process():
    """解释生成过程"""
    print("\n" + "=" * 60)
    print("【4. 生成过程详解】")
    print("=" * 60)

    print("""
GPT 生成文本的过程（自回归）：

步骤 1: 输入 "今"
       → 模型预测下一个字的概率分布
       → 采样得到 "天"

步骤 2: 输入 "今天"
       → 模型预测下一个字的概率分布
       → 采样得到 "天"

步骤 3: 输入 "今天天"
       → 模型预测下一个字的概率分布
       → 采样得到 "气"

步骤 4: 输入 "今天天气"
       → ... 继续 ...

关键点：
1. 每次只生成一个 token
2. 生成的 token 会拼接到输入中
3. 循环直到达到最大长度或遇到结束符

这就是"自回归"的含义：输出又成为下一步的输入。
    """)


def main():
    print("=" * 60)
    print("GPT 文本生成演示")
    print("=" * 60)

    demo_untrained_generation()
    demo_sampling_strategies()
    demo_generation_params()
    explain_generation_process()

    print("\n" + "=" * 60)
    print("总结")
    print("=" * 60)
    print("""
文本生成要点：

1. 自回归：每次生成一个 token，拼接到输入
2. 温度：控制随机性（0.7-1.0 推荐）
3. Top-k：过滤低概率 token（40 推荐）
4. 未训练的模型输出是乱码

下一步：训练模型！
    """)


if __name__ == "__main__":
    main()
