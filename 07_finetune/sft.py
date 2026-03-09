"""
=================================================================================
第13课: 监督微调 (Supervised Fine-Tuning, SFT)
=================================================================================

什么是微调？
----------
微调是在预训练模型的基础上，用特定任务的数据继续训练。

预训练 vs 微调：
- 预训练：在海量文本上学习"语言的规律"
- 微调：在特定任务数据上学习"如何完成任务"

SFT 的数据格式：
    输入: "用户: 请介绍一下红楼梦\n助手: "
    目标: "用户: 请介绍一下红楼梦\n助手: 红楼梦是..."

运行方式：
    python sft.py                    # 查看概念说明
    python sft.py --train            # 执行演示训练
    python sft.py --train --checkpoint path  # 从检查点训练
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import os
import sys
import argparse
import pickle

# 添加路径
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, '05_gpt_model'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, '06_pretrain'))

from model import GPT, GPTConfig


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

    @classmethod
    def load(cls, path):
        """从文件加载分词器"""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        tokenizer = cls()
        tokenizer.chars = data['chars']
        tokenizer.char_to_id = data['char_to_id']
        tokenizer.id_to_char = data['id_to_char']
        tokenizer.vocab_size = data['vocab_size']
        return tokenizer

    def save(self, path):
        """保存分词器"""
        with open(path, 'wb') as f:
            pickle.dump({
                'chars': self.chars,
                'char_to_id': self.char_to_id,
                'id_to_char': self.id_to_char,
                'vocab_size': self.vocab_size
            }, f)


class SFTDataset(Dataset):
    """
    SFT 数据集

    关键：只在"输出"部分计算损失，输入部分的标签设为 -100
    """

    def __init__(self, data, tokenizer, max_length=256):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # 构建输入文本
        instruction = item.get('instruction', '')
        input_text = item.get('input', '')
        output = item.get('output', '')

        # 组合成对话格式
        prompt = f"用户: {instruction}\n{input_text}\n助手: "
        full_text = prompt + output

        # 编码
        prompt_ids = self.tokenizer.encode(prompt)
        full_ids = self.tokenizer.encode(full_text)

        # 截断
        if len(full_ids) > self.max_length:
            full_ids = full_ids[:self.max_length]

        # 创建标签（输入部分设为 -100，只计算输出的损失）
        labels = full_ids.copy()
        prompt_len = min(len(prompt_ids), len(full_ids))
        labels[:prompt_len] = [-100] * prompt_len

        # 创建输入（右移一位）
        input_ids = full_ids[:-1]
        labels = labels[1:]

        return torch.tensor(input_ids, dtype=torch.long), torch.tensor(labels, dtype=torch.long)


def collate_fn(batch):
    """自定义批次处理，处理变长序列"""
    inputs, labels = zip(*batch)

    # 找到最大长度
    max_len = max(len(x) for x in inputs)

    # 填充
    padded_inputs = []
    padded_labels = []
    for x, y in zip(inputs, labels):
        pad_len = max_len - len(x)
        padded_inputs.append(F.pad(x, (0, pad_len), value=0))
        padded_labels.append(F.pad(y, (0, pad_len), value=-100))

    return torch.stack(padded_inputs), torch.stack(padded_labels)


def train_sft(model, train_data, tokenizer, epochs=3, batch_size=4, lr=5e-5, device='cpu'):
    """执行 SFT 训练"""
    print("=" * 70)
    print("SFT 训练")
    print("=" * 70)

    # 创建数据集
    dataset = SFTDataset(train_data, tokenizer)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    print(f"训练样本数: {len(dataset)}")
    print(f"批次大小: {batch_size}")
    print(f"学习率: {lr}")
    print(f"训练轮数: {epochs}")

    # 优化器（使用较小的学习率）
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    # 学习率调度器
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs * len(dataloader))

    model.train()
    model.to(device)

    global_step = 0

    print(f"\n{'Epoch':>6} | {'Step':>6} | {'Loss':>10} | {'LR':>10}")
    print("-" * 50)

    for epoch in range(epochs):
        epoch_loss = 0
        for batch_idx, (inputs, labels) in enumerate(dataloader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            # 前向传播
            logits, loss = model(inputs, labels)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            scheduler.step()

            global_step += 1
            epoch_loss += loss.item()

            if batch_idx % 5 == 0:
                current_lr = scheduler.get_last_lr()[0]
                print(f"{epoch+1:>6} | {batch_idx:>6} | {loss.item():>10.4f} | {current_lr:>10.2e}")

        avg_loss = epoch_loss / len(dataloader)
        print(f"\nEpoch {epoch+1} 完成，平均损失: {avg_loss:.4f}\n")

    print("=" * 70)
    print("SFT 训练完成!")
    print("=" * 70)

    return model


def generate_response(model, tokenizer, instruction, input_text="", max_tokens=100, device='cpu'):
    """生成回答"""
    model.eval()

    prompt = f"用户: {instruction}\n{input_text}\n助手: "
    idx = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long).to(device)

    with torch.no_grad():
        output = model.generate(idx, max_new_tokens=max_tokens, temperature=0.7, top_k=40)

    generated = tokenizer.decode(output[0].tolist())
    # 只返回助手回答部分
    response = generated[len(prompt):]
    return response


def demo_sft():
    """演示 SFT 训练"""
    print("=" * 70)
    print("SFT 训练演示")
    print("=" * 70)

    # 设备 (使用 CPU 避免 MPS 兼容性问题)
    device = 'cpu'
    print(f"使用设备: {device}")

    # 模拟 SFT 训练数据
    sft_data = [
        {"instruction": "续写", "input": "林黛玉", "output": "听了这话，不觉心动，眼圈儿一红，便背过脸去。"},
        {"instruction": "续写", "input": "贾宝玉", "output": "见她这般光景，心中更加疑惑，忙上前问道：妹妹这是怎么说？"},
        {"instruction": "续写", "input": "王熙凤", "output": "笑道：你们两个又在这里闹什么？快跟我去老太太那里。"},
        {"instruction": "续写", "input": "薛宝钗", "output": "微微一笑，说道：这也难怪，本就是多心的人，何苦再来惹她。"},
        {"instruction": "续写", "input": "晴雯", "output": "冷笑道：什么稀罕东西，也值得这样大惊小怪的。"},
        {"instruction": "续写", "input": "袭人", "output": "忙劝道：姑娘休要如此，仔细老太太知道了不依。"},
        {"instruction": "续写", "input": "探春", "output": "叹道：咱们家里的事，说起来也实在叫人寒心。"},
        {"instruction": "续写", "input": "湘云", "output": "拍手笑道：这才是正经话，我就爱听你这样说。"},
    ]

    # 构建所有文本用于创建分词器（包含特殊标记字符）
    special_chars = "用户助手：:\n "  # 特殊字符
    all_text = special_chars
    for item in sft_data:
        all_text += item['instruction'] + item['input'] + item['output']

    # 创建分词器
    tokenizer = SimpleTokenizer(all_text)
    print(f"词汇表大小: {tokenizer.vocab_size}")

    # 创建模型（小模型用于演示）
    config = GPTConfig(
        vocab_size=tokenizer.vocab_size,
        block_size=256,
        n_layer=4,
        n_head=4,
        n_embd=128,
        dropout=0.1
    )
    model = GPT(config)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {n_params:,} ({n_params/1e6:.2f}M)")

    # 训练前测试
    print("\n" + "=" * 70)
    print("训练前生成测试")
    print("=" * 70)
    test_prompts = [
        ("续写", "林黛玉"),
        ("续写", "贾宝玉"),
    ]
    for instruction, input_text in test_prompts:
        response = generate_response(model, tokenizer, instruction, input_text, device=device)
        print(f"输入: {instruction} - {input_text}")
        print(f"输出: {response}\n")

    # 执行训练
    model = train_sft(
        model, sft_data, tokenizer,
        epochs=10,  # 演示用，多训练几轮
        batch_size=2,
        lr=1e-3,  # 小模型可以用较高学习率
        device=device
    )

    # 训练后测试
    print("\n" + "=" * 70)
    print("训练后生成测试")
    print("=" * 70)
    for instruction, input_text in test_prompts:
        response = generate_response(model, tokenizer, instruction, input_text, device=device)
        print(f"输入: {instruction} - {input_text}")
        print(f"输出: {response}\n")

    return model, tokenizer


def train_from_checkpoint(checkpoint_path, sft_data_path=None):
    """从检查点加载模型并执行 SFT"""
    print("=" * 70)
    print("从检查点加载模型进行 SFT")
    print("=" * 70)

    # 设备
    device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")

    # 加载检查点
    print(f"加载检查点: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # 获取配置
    model_config = checkpoint.get('config')
    if model_config is None:
        model_config = GPTConfig(vocab_size=3000, block_size=256, n_layer=6, n_head=6, n_embd=192)

    # 创建模型
    model = GPT(model_config)
    model.load_state_dict(checkpoint['model'])
    model.to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {n_params:,} ({n_params/1e6:.2f}M)")

    # 加载分词器
    tokenizer_path = os.path.join(os.path.dirname(checkpoint_path), 'tokenizer.pkl')
    if os.path.exists(tokenizer_path):
        tokenizer = SimpleTokenizer.load(tokenizer_path)
        print(f"分词器已加载，词汇表大小: {tokenizer.vocab_size}")
    else:
        print("警告: 未找到分词器，请确保检查点目录中有 tokenizer.pkl")
        return None, None

    # 加载 SFT 数据
    if sft_data_path and os.path.exists(sft_data_path):
        import json
        with open(sft_data_path, 'r', encoding='utf-8') as f:
            sft_data = json.load(f)
        print(f"加载 SFT 数据: {len(sft_data)} 条")
    else:
        print("使用内置示例数据")
        sft_data = [
            {"instruction": "续写", "input": "林黛玉", "output": "听了这话，不觉心动。"},
            {"instruction": "续写", "input": "贾宝玉", "output": "见她这般光景，心中更加疑惑。"},
        ]

    # 执行训练
    model = train_sft(model, sft_data, tokenizer, epochs=3, batch_size=4, lr=5e-5, device=device)

    # 保存微调后的模型
    output_path = os.path.join(os.path.dirname(checkpoint_path), 'model_sft.pt')
    torch.save({
        'model': model.state_dict(),
        'config': model_config,
    }, output_path)
    print(f"微调后的模型已保存到: {output_path}")

    return model, tokenizer


def explain_sft_concepts():
    """解释 SFT 概念"""
    print("""
================================================================================
SFT (Supervised Fine-Tuning) 核心概念
================================================================================

1. 训练流程：
   预训练 → SFT → RLHF/DPO

2. SFT 与预训练的区别：
   ┌──────────────┬───────────────────┬────────────────────┐
   │     方面     │      预训练       │        SFT         │
   ├──────────────┼───────────────────┼────────────────────┤
   │ 数据格式     │ 纯文本            │ 指令-回答对        │
   │ 损失计算     │ 所有位置          │ 只在输出部分       │
   │ 学习率       │ ~1e-3             │ ~1e-5 到 5e-5      │
   │ 数据量       │ 海量（TB级）      │ 中等（万级）       │
   │ 训练时间     │ 周级              │ 小时到天级         │
   └──────────────┴───────────────────┴────────────────────┘

3. 关键技巧：
   - 使用 -100 标签忽略 prompt 部分的损失
   - 学习率要比预训练小 10-100 倍
   - 数据质量比数量更重要
   - 可以使用 LoRA 减少显存占用

4. 常见问题：
   - 灾难性遗忘：微调太久会忘记预训练知识
   - 过拟合：数据太少会导致模型死记硬背
   - 风格漂移：模型可能学会不良回答模式
================================================================================
    """)


def main():
    parser = argparse.ArgumentParser(description='SFT 微调训练')
    parser.add_argument('--train', action='store_true', help='执行真实训练')
    parser.add_argument('--checkpoint', type=str, default=None, help='预训练模型检查点路径')
    parser.add_argument('--data', type=str, default=None, help='SFT 数据文件路径 (JSON)')
    parser.add_argument('--epochs', type=int, default=3, help='训练轮数')
    parser.add_argument('--lr', type=float, default=5e-5, help='学习率')
    args = parser.parse_args()

    print("=" * 70)
    print("第13课: 监督微调 (SFT)")
    print("=" * 70)

    if args.train:
        if args.checkpoint:
            train_from_checkpoint(args.checkpoint, args.data)
        else:
            demo_sft()
    else:
        explain_sft_concepts()
        print("\n提示: 使用 --train 参数执行真实训练")
        print("  python sft.py --train                          # 运行演示训练")
        print("  python sft.py --train --checkpoint model.pt    # 从检查点训练")


if __name__ == "__main__":
    main()
