#!/usr/bin/env python3
"""
从零开始训练 GPT - 使用经过验证的配置

配置基于：
- nanoGPT (Karpathy)
- Chinchilla Scaling Laws

使用方法:
    python train.py                      # 默认 mini 配置
    python train.py --config micro        # 更小更快
    python train.py --config small        # 更大更好
    python train.py --max_iters 10000    # 自定义迭代数
"""

import os
import sys
import time
import math
import argparse

import torch

# 添加路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import GPTConfig, TrainConfig, get_config, CONFIGS

# 导入模型
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '05_gpt_model'))
from model import GPT as GPTModel


class CharTokenizer:
    """字符级分词器"""

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

    def save(self, path):
        import pickle
        with open(path, 'wb') as f:
            pickle.dump({
                'chars': self.chars,
                'char_to_id': self.char_to_id,
                'id_to_char': self.id_to_char,
                'vocab_size': self.vocab_size
            }, f)


def get_lr(it, learning_rate, warmup_iters, lr_decay_iters, min_lr):
    """
    学习率调度 (nanoGPT 使用的 Cosine with Warmup)

    基于: https://github.com/karpathy/nanoGPT
    """
    # 1) Warmup 阶段：线性增加
    if it < warmup_iters:
        return learning_rate * it / warmup_iters

    # 2) 超过衰减期：使用最小学习率
    if it > lr_decay_iters:
        return min_lr

    # 3) Cosine 衰减
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff goes from 1 to 0
    return min_lr + coeff * (learning_rate - min_lr)


def get_batch(data, block_size, batch_size, device):
    """获取训练批次"""
    ix = torch.randint(0, len(data) - block_size - 1, (batch_size,))
    x = torch.stack([torch.tensor(data[i:i+block_size], dtype=torch.long) for i in ix])
    y = torch.stack([torch.tensor(data[i+1:i+block_size+1], dtype=torch.long) for i in ix])
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss(model, data, block_size, batch_size, device, eval_iters=100):
    """评估损失"""
    model.eval()
    losses = []
    for _ in range(eval_iters):
        X, Y = get_batch(data, block_size, batch_size, device)
        _, loss = model(X, Y)
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


def train(config_name='mini', max_iters=None):
    """训练函数"""

    # 获取配置
    config = get_config(config_name)
    model_cfg = config['model']
    train_cfg = config['train']

    # 覆盖迭代数
    if max_iters is not None:
        train_cfg['max_iters'] = max_iters

    # 设备
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"使用设备: {device}")
    print(f"配置: {config_name}")

    # 读取数据
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
    data_path = os.path.join(data_dir, 'wikitext_103.txt')
    if not os.path.exists(data_path):
        data_path = os.path.join(data_dir, 'wikitext.txt')
    print(f"数据文件: {data_path}")

    with open(data_path, 'r', encoding='utf-8') as f:
        text = f.read()

    print(f"文本长度: {len(text):,} 字符")

    # 分词器
    tokenizer = CharTokenizer(text)
    print(f"词汇表大小: {tokenizer.vocab_size}")

    # 编码
    data = tokenizer.encode(text)
    print(f"Token 数量: {len(data):,}")

    # 创建模型
    model_config = GPTConfig(
        vocab_size=tokenizer.vocab_size,
        block_size=model_cfg['block_size'],
        n_layer=model_cfg['n_layer'],
        n_head=model_cfg['n_head'],
        n_embd=model_cfg['n_embd'],
    )
    model = GPTModel(model_config)
    model.to(device)

    # 计算参数量
    n_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数: {n_params:,} ({n_params/1e6:.2f}M)")

    # 计算理论最优训练量 (Chinchilla: 20 tokens/参数)
    optimal_tokens = n_params * 20
    optimal_epochs = optimal_tokens / len(data)
    print(f"Chinchilla 最优: {optimal_tokens/1e6:.1f}M tokens = {optimal_epochs:.0f} epochs")

    # 优化器 (nanoGPT 推荐设置)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg['learning_rate'],
        betas=(0.9, 0.95),
        weight_decay=0.1,
    )

    # 训练循环
    print(f"\n开始训练 ({train_cfg['max_iters']} 迭代)...")
    print("=" * 70)
    print(f"{'迭代':>8} | {'损失':>8} | {'学习率':>10} | {'时间':>8} | {'Tokens':>12}")
    print("=" * 70)

    best_loss = float('inf')
    t0 = time.time()

    for iter in range(train_cfg['max_iters']):

        # 学习率调度
        lr = get_lr(
            iter,
            train_cfg['learning_rate'],
            100,  # warmup
            train_cfg['max_iters'],  # decay 到最后
            train_cfg['learning_rate'] / 10,  # min_lr
        )
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # 获取批次
        X, Y = get_batch(data, model_cfg['block_size'], train_cfg['batch_size'], device)

        # 前向传播
        _, loss = model(X, Y)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()

        # 梯度裁剪 (nanoGPT 推荐 1.0)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()

        # 记录最佳
        if loss.item() < best_loss:
            best_loss = loss.item()

        # 打印进度
        if iter % 100 == 0 or iter == train_cfg['max_iters'] - 1:
            t1 = time.time()
            dt = t1 - t0
            tokens_seen = iter * train_cfg['batch_size'] * model_cfg['block_size']
            print(f"{iter:>8} | {loss.item():8.4f} | {lr:>10.2e} | {dt:>8.1f}s | {tokens_seen/1e6:>10.2f}M")

        # 定期评估
        if iter % 500 == 0 and iter > 0:
            val_loss = estimate_loss(model, data, model_cfg['block_size'], train_cfg['batch_size'], device, 50)
            print(f"{'验证':>8} | {val_loss:8.4f} | {'':>10} | {'':>8} | {'':>12}")

    # 训练完成
    print("=" * 70)
    total_time = time.time() - t0
    print(f"训练完成! 最佳损失: {best_loss:.4f}, 总时间: {total_time:.1f}s")

    # 保存
    checkpoint_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)

    torch.save({
        'model': model.state_dict(),
        'config': model_config,
        'train_config': {'iter': train_cfg['max_iters'], 'best_loss': best_loss},
    }, os.path.join(checkpoint_dir, 'model.pt'))

    tokenizer.save(os.path.join(checkpoint_dir, 'tokenizer.pkl'))
    print(f"模型保存到: checkpoints/model.pt")

    # 测试生成
    print("\n" + "=" * 70)
    print("测试生成")
    print("=" * 70)
    model.eval()

    prompts = ["The", "In the beginning", "Once upon a time"]
    for prompt in prompts:
        idx = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long).to(device)
        with torch.no_grad():
            output = model.generate(idx, max_new_tokens=50, temperature=0.8, top_k=40)
        generated = tokenizer.decode(output[0].tolist())
        print(f"\nPrompt: '{prompt}'")
        print(f"生成: '{generated}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='GPT 训练')
    parser.add_argument('--config', type=str, default='mini',
                        choices=list(CONFIGS.keys()),
                        help='模型配置')
    parser.add_argument('--max_iters', type=int, default=None,
                        help='训练迭代数')
    args = parser.parse_args()

    train(args.config, args.max_iters)
