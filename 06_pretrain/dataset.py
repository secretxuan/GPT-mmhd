"""
=================================================================================
第11课: 数据集准备
=================================================================================

训练 GPT 需要准备数据集。这节课学习如何把文本变成训练数据。

训练数据的格式
-------------
GPT 的训练目标是"预测下一个字符"：

    输入 (x):  "红楼梦"  →  目标 (y):  "楼梦真"
    输入 (x):  "楼梦真"  →  目标 (y):  "梦真是"
    ...

也就是说：
- 输入是文本的 [0:n]
- 目标是文本的 [1:n+1]
- 每个位置都预测下一个字符

数据集的划分
-----------
通常分为三部分：
- 训练集 (Train): 90% - 用于训练模型
- 验证集 (Val): 5% - 用于调参、选择模型
- 测试集 (Test): 5% - 用于最终评估

运行方式：python dataset.py
"""

import torch
from torch.utils.data import Dataset, DataLoader
import os
import pickle


class SimpleTokenizer:
    """简单的字符级分词器"""

    def __init__(self, text=None):
        self.chars = []
        self.char_to_id = {}
        self.id_to_char = {}
        self.vocab_size = 0
        if text is not None:
            self.train(text)

    def train(self, text):
        """从文本构建词汇表"""
        self.chars = sorted(list(set(text)))
        self.char_to_id = {ch: i for i, ch in enumerate(self.chars)}
        self.id_to_char = {i: ch for i, ch in enumerate(self.chars)}
        self.vocab_size = len(self.chars)

    def encode(self, text):
        return [self.char_to_id[ch] for ch in text]

    def decode(self, ids):
        return ''.join([self.id_to_char[i] for i in ids])


class TextDataset(Dataset):
    """
    文本数据集

    把长文本切成固定长度的片段，用于训练
    """

    def __init__(self, data, block_size):
        """
        Args:
            data: token ID 列表
            block_size: 每个样本的长度
        """
        self.data = data
        self.block_size = block_size

        # 计算可以切出多少个样本
        # 每个样本需要 block_size + 1 个 token（因为目标是输入后移一位）
        self.n_samples = len(data) // (block_size + 1)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        """
        获取一个样本

        Returns:
            x: 输入 token IDs, [block_size]
            y: 目标 token IDs, [block_size]
        """
        # 起始位置
        start = idx * (self.block_size + 1)
        end = start + self.block_size + 1

        # 切片
        chunk = self.data[start:end]

        # 输入和目标
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)

        return x, y


def prepare_data(text_path, block_size=256, train_ratio=0.9):
    """
    准备训练数据

    Args:
        text_path: 文本文件路径
        block_size: 序列长度
        train_ratio: 训练集比例

    Returns:
        train_dataset, val_dataset, tokenizer
    """
    # 读取文本
    with open(text_path, 'r', encoding='utf-8') as f:
        text = f.read()

    print(f"文本长度: {len(text)} 字符")

    # 创建分词器
    tokenizer = SimpleTokenizer()
    tokenizer.train(text)

    # 编码
    data = tokenizer.encode(text)
    print(f"编码后长度: {len(data)} tokens")

    # 划分训练集和验证集
    split = int(len(data) * train_ratio)
    train_data = data[:split]
    val_data = data[split:]

    print(f"训练集: {len(train_data)} tokens")
    print(f"验证集: {len(val_data)} tokens")

    # 创建数据集
    train_dataset = TextDataset(train_data, block_size)
    val_dataset = TextDataset(val_data, block_size)

    print(f"训练样本数: {len(train_dataset)}")
    print(f"验证样本数: {len(val_dataset)}")

    return train_dataset, val_dataset, tokenizer


def demo_dataset():
    """演示数据集"""
    print("=" * 60)
    print("【1. 数据集演示】")
    print("=" * 60)

    # 使用红楼梦数据
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'hongloumeng.txt')

    if not os.path.exists(data_path):
        print(f"数据文件不存在: {data_path}")
        return

    train_dataset, val_dataset, tokenizer = prepare_data(data_path, block_size=64)

    # 查看一个样本
    x, y = train_dataset[0]
    print(f"\n样本示例:")
    print(f"  输入 x: {x[:20].tolist()}...")
    print(f"  目标 y: {y[:20].tolist()}...")
    print(f"  输入解码: '{tokenizer.decode(x[:20].tolist())}...'")
    print(f"  目标解码: '{tokenizer.decode(y[:20].tolist())}...'")


def demo_dataloader():
    """演示 DataLoader"""
    print("\n" + "=" * 60)
    print("【2. DataLoader 演示】")
    print("=" * 60)

    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'hongloumeng.txt')

    if not os.path.exists(data_path):
        print(f"数据文件不存在: {data_path}")
        return

    train_dataset, _, _ = prepare_data(data_path, block_size=64)

    # 创建 DataLoader
    batch_size = 4
    dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,  # 打乱顺序
        num_workers=0,  # 数据加载线程数
        drop_last=True  # 丢弃最后不完整的 batch
    )

    print(f"\nDataLoader 配置:")
    print(f"  batch_size: {batch_size}")
    print(f"  总 batch 数: {len(dataloader)}")

    # 获取一个 batch
    x, y = next(iter(dataloader))
    print(f"\n一个 batch:")
    print(f"  x 形状: {x.shape}")
    print(f"  y 形状: {y.shape}")


def explain_batching():
    """解释批处理"""
    print("\n" + "=" * 60)
    print("【3. 批处理 (Batching)】")
    print("=" * 60)

    print("""
为什么用 batch？

1. 效率：GPU 擅长并行处理
   - 一次处理 32 个样本比逐个处理快得多

2. 梯度更稳定
   - 多个样本的梯度平均，减少噪声

3. 内存限制
   - batch_size 越大，显存占用越多
   - 需要根据 GPU 内存选择合适的 batch_size

常见的 batch_size：
- 大模型训练：32-256
- 小模型/学习：8-32

如果显存不够，可以：
- 减小 batch_size
- 使用梯度累积（多个小 batch 的梯度累积后再更新）
    """)


def demo_gradient_accumulation():
    """演示梯度累积"""
    print("\n" + "=" * 60)
    print("【4. 梯度累积】")
    print("=" * 60)

    print("""
梯度累积技巧：模拟大 batch_size

问题：显存只够 batch_size=4，但想要 batch_size=32 的效果

解决：
    accumulation_steps = 32 / 4 = 8

    for i, batch in enumerate(dataloader):
        loss = model(batch)

        # 归一化损失
        loss = loss / accumulation_steps
        loss.backward()

        # 每 8 步更新一次
        if (i + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

效果：相当于 batch_size=32，但显存只需要 batch_size=4

代价：训练速度变慢（8 倍）
    """)


def save_processed_data(data, tokenizer, output_dir):
    """保存处理后的数据"""
    os.makedirs(output_dir, exist_ok=True)

    # 保存数据
    with open(os.path.join(output_dir, 'data.bin'), 'wb') as f:
        pickle.dump(data, f)

    # 保存分词器
    tokenizer.save(os.path.join(output_dir, 'tokenizer.pkl'))

    print(f"数据已保存到: {output_dir}")


def main():
    print("=" * 60)
    print("第11课: 数据集准备")
    print("=" * 60)

    demo_dataset()
    demo_dataloader()
    explain_batching()
    demo_gradient_accumulation()

    print("\n" + "=" * 60)
    print("总结")
    print("=" * 60)
    print("""
数据准备要点：

1. 数据格式：输入是文本 [0:n]，目标是 [1:n+1]
2. 划分：训练集 90%，验证集 10%
3. 批处理：提高效率，稳定梯度
4. 梯度累积：显存不够时的解决方案

下一步：开始训练！
    """)


if __name__ == "__main__":
    main()
