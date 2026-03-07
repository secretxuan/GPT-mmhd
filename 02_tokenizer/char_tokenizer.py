"""
=================================================================================
第4课: 字符级分词器 (Character Tokenizer)
=================================================================================

什么是分词器？
------------
分词器的作用是把"文本"转成"数字"，让神经网络能处理。

    "你好世界" → [100, 101, 102, 103] → 神经网络

为什么需要分词器？
----------------
- 神经网络只能处理数字，不能直接处理文字
- 分词器就是文字和数字之间的"翻译官"

分词器的类型：
1. 字符级 (Character-level): 每个字符一个编号（我们要实现的，最简单）
2. 词级 (Word-level): 每个词一个编号
3. 子词级 (Subword/BBPE): 现代GPT使用的，介于词和字符之间

运行方式：python char_tokenizer.py
"""

import os
import pickle
from typing import List, Dict


class CharTokenizer:
    """
    字符级分词器

    功能：
    - encode: 文本 → 数字列表
    - decode: 数字列表 → 文本

    原理：
    1. 遍历文本，统计所有出现过的字符
    2. 给每个字符分配一个唯一的 ID（从0开始）
    3. encode: 查表，把字符转成 ID
    4. decode: 反向查表，把 ID 转成字符
    """

    def __init__(self):
        self.char_to_id: Dict[str, int] = {}  # 字符 → ID 的映射
        self.id_to_char: Dict[int, str] = {}  # ID → 字符的映射
        self.vocab_size: int = 0  # 词汇表大小（总共多少个不同的字符）

    def train(self, text: str) -> None:
        """
        从文本中构建词汇表

        Args:
            text: 训练文本
        """
        # 找出所有出现过的字符（去重）
        chars = sorted(list(set(text)))

        # 构建 字符→ID 映射
        self.char_to_id = {ch: i for i, ch in enumerate(chars)}

        # 构建 ID→字符 映射（反向）
        self.id_to_char = {i: ch for i, ch in enumerate(chars)}

        # 记录词汇表大小
        self.vocab_size = len(chars)

        print(f"词汇表大小: {self.vocab_size}")
        print(f"前20个字符: {chars[:20]}")

    def encode(self, text: str) -> List[int]:
        """
        把文本编码成数字列表

        Args:
            text: 要编码的文本

        Returns:
            数字列表

        Example:
            >>> tokenizer.encode("你好")
            [100, 101]
        """
        return [self.char_to_id[ch] for ch in text]

    def decode(self, ids: List[int]) -> str:
        """
        把数字列表解码成文本

        Args:
            ids: 数字列表

        Returns:
            解码后的文本

        Example:
            >>> tokenizer.decode([100, 101])
            "你好"
        """
        return ''.join([self.id_to_char[i] for i in ids])

    def save(self, path: str) -> None:
        """保存分词器到文件"""
        data = {
            'char_to_id': self.char_to_id,
            'id_to_char': self.id_to_char,
            'vocab_size': self.vocab_size
        }
        with open(path, 'wb') as f:
            pickle.dump(data, f)
        print(f"分词器已保存到: {path}")

    @classmethod
    def load(cls, path: str) -> 'CharTokenizer':
        """从文件加载分词器"""
        with open(path, 'rb') as f:
            data = pickle.load(f)

        tokenizer = cls()
        tokenizer.char_to_id = data['char_to_id']
        tokenizer.id_to_char = data['id_to_char']
        tokenizer.vocab_size = data['vocab_size']
        print(f"分词器已加载，词汇表大小: {tokenizer.vocab_size}")
        return tokenizer


def demo_basic_usage():
    """演示基本用法"""
    print("=" * 60)
    print("【1. 基本用法】")
    print("=" * 60)

    # 训练文本
    text = "你好世界！这是一段测试文本。Hello, World!"

    # 创建分词器并训练
    tokenizer = CharTokenizer()
    tokenizer.train(text)

    # 编码
    test_text = "你好世界"
    encoded = tokenizer.encode(test_text)
    print(f"\n编码示例:")
    print(f"  原文: '{test_text}'")
    print(f"  编码: {encoded}")

    # 解码
    decoded = tokenizer.decode(encoded)
    print(f"\n解码示例:")
    print(f"  编码: {encoded}")
    print(f"  解码: '{decoded}'")

    # 验证：编码再解码应该得到原文
    assert test_text == decoded, "编码解码不一致！"
    print("\n✓ 编码解码验证通过！")


def demo_with_novel():
    """演示用小说文本训练分词器"""
    print("\n" + "=" * 60)
    print("【2. 用小说文本训练】")
    print("=" * 60)

    # 模拟一段小说文本
    novel_text = """
    第一回 甄士隐梦幻识通灵 贾雨村风尘怀闺秀

    此开卷第一回也。作者自云：因曾历过一番梦幻之后，故将真事隐去，
    而借"通灵"之说，撰此《石头记》一书也。故曰"甄士隐"云云。

    但书中所记何事何人？自又云："今风尘碌碌，一事无成，忽念及当日
    所有之女子，一一细考较去，觉其行止见识，皆出于我之上。
    """

    # 创建分词器
    tokenizer = CharTokenizer()
    tokenizer.train(novel_text)

    print(f"\n词汇表（所有不同的字符）:")
    vocab = sorted(tokenizer.char_to_id.keys())
    print(f"  共 {len(vocab)} 个字符")

    # 测试编码（使用训练数据中存在的字符）
    test_text = "甄士隐"
    encoded = tokenizer.encode(test_text)
    print(f"\n测试编码:")
    print(f"  '{test_text}' → {encoded}")

    # 逐字符显示
    print(f"\n逐字符分解:")
    for ch in test_text:
        print(f"  '{ch}' → {tokenizer.char_to_id[ch]}")


def demo_gpt_tokenizer_context():
    """解释 GPT 使用的分词器"""
    print("\n" + "=" * 60)
    print("【3. GPT 使用的分词器 (BPE)】")
    print("=" * 60)

    print("""
我们实现的是"字符级分词器"，这是最简单的。

现代 GPT 使用的是 BPE (Byte Pair Encoding) 分词器：
- 它不是按字符切分，而是按"常见片段"切分
- 例如："university" 可能被分成 ["un", "ivers", "ity"]
- 这样可以让模型更快学习，词汇表也更小

对比：
┌─────────────────┬────────────────────────┐
│     方法        │        特点            │
├─────────────────┼────────────────────────┤
│ 字符级 (我们)   │ 最简单，但序列很长     │
│ 词级            │ 需要先分词，词汇表巨大 │
│ BPE (GPT)       │ 平衡，现代标准做法     │
└─────────────────┴────────────────────────┘

字符级分词器的优点：
1. 实现简单，容易理解
2. 不需要分词，适合中文
3. 可以生成任何字符

缺点：
1. 序列长度很长（一个汉字 = 一个 token）
2. 模型需要学习的"跨度"更大

对于学习 GPT 原理，字符级分词器完全够用！
    """)


def demo_tokenizer_in_training():
    """演示分词器在训练中的使用"""
    print("\n" + "=" * 60)
    print("【4. 分词器在训练中的使用】")
    print("=" * 60)

    # 模拟训练数据
    text = "林黛玉听了这话，不觉心动，便问道：你是何人？"

    # 创建分词器
    tokenizer = CharTokenizer()
    tokenizer.train(text)

    # 把整个文本转成数字
    data = tokenizer.encode(text)

    print(f"原文: {text}")
    print(f"编码后: {data}")
    print(f"长度: {len(data)} 个 token")

    # 在训练中，我们这样组织数据：
    # 输入: data[0:n]    预测: data[1:n+1]
    # 也就是"预测下一个字符"

    print("\n训练数据组织方式:")
    block_size = 8  # 每次输入8个字符

    for i in range(len(data) - block_size):
        x = data[i:i+block_size]
        y = data[i+1:i+block_size+1]

        print(f"\n样本 {i+1}:")
        print(f"  输入 x: {x}")
        print(f"  输入解码: '{tokenizer.decode(x)}'")
        print(f"  目标 y: {y}")
        print(f"  目标解码: '{tokenizer.decode(y)}'")
        print("  解释: 给定输入，预测目标（每个位置预测下一个字符）")

        if i >= 1:  # 只展示前几个
            break

    print("""
... 以此类推

这就是 GPT 的训练方式：
- 给定一段文本的前 N 个字符
- 让模型预测第 2 到 N+1 个字符
- 本质是"预测下一个字符"的任务
    """)


def main():
    print("=" * 60)
    print("第4课: 字符级分词器")
    print("=" * 60)

    # 1. 基本用法
    demo_basic_usage()

    # 2. 用小说文本训练
    demo_with_novel()

    # 3. GPT 使用的分词器
    demo_gpt_tokenizer_context()

    # 4. 分词器在训练中的使用
    demo_tokenizer_in_training()

    print("\n" + "=" * 60)
    print("总结")
    print("=" * 60)
    print("""
分词器的作用：
1. 把文本转成数字（encode）
2. 把数字转回文本（decode）

字符级分词器：
- 最简单，适合学习
- 每个字符一个 ID
- 中文：一个汉字 = 一个 token

下一步：学习 GPT 的核心机制 —— 注意力！
    """)


if __name__ == "__main__":
    main()
