"""
=================================================================================
第9课: 完整的 Transformer Block
=================================================================================

现在我们把所有组件组装起来！

Transformer Block 的完整结构
--------------------------
每个 Block 包含：
1. LayerNorm + 因果自注意力 + 残差连接
2. LayerNorm + MLP + 残差连接

结构图（Pre-Norm 风格，GPT 使用）：

    输入 x
       │
       ├──────────────────────────────┐
       ↓                              │
    LayerNorm                         │
       ↓                              │
    因果自注意力                      │
       ↓                              │
    Dropout                           │
       ↓                              │
      (+) ←───────────────────────────┘ 残差连接
       │
       ├──────────────────────────────┐
       ↓                              │
    LayerNorm                         │
       ↓                              │
    MLP                               │
       ↓                              │
    Dropout                           │
       ↓                              │
      (+) ←───────────────────────────┘ 残差连接
       │
    输出

残差连接的作用
------------
x + sublayer(x) 而不是 sublayer(x)

优点：
1. 梯度可以直接"跳过"子层，缓解梯度消失
2. 允许训练更深的网络
3. 每个块可以"选择"使用或不使用子层

运行方式：python block.py
"""

import torch
import torch.nn as nn
import math

# 导入之前实现的组件（假设在同一目录）
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from mlp import MLP


class CausalSelfAttention(nn.Module):
    """因果自注意力（从之前的课程复制）"""

    def __init__(self, embed_dim, num_heads, dropout=0.0):
        super().__init__()
        assert embed_dim % num_heads == 0

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Q、K、V 投影
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)

        # 输出投影
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        # Dropout
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

    def forward(self, x):
        batch_size, seq_len, _ = x.shape

        # 计算 Q、K、V
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        # 重塑为多头形式
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # 计算注意力分数
        scores = Q @ K.transpose(-2, -1) / math.sqrt(self.head_dim)

        # 应用因果掩码
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device))
        scores = scores.masked_fill(causal_mask == 0, float('-inf'))

        # Softmax
        attention_weights = torch.softmax(scores, dim=-1)
        attention_weights = self.attn_dropout(attention_weights)

        # 加权求和
        output = attention_weights @ V

        # 合并多头
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)

        # 输出投影
        output = self.resid_dropout(self.out_proj(output))

        return output


class TransformerBlock(nn.Module):
    """
    完整的 Transformer Block

    这是 GPT 的基本构建单元
    """

    def __init__(self, embed_dim, num_heads, dropout=0.0):
        super().__init__()

        # Layer Norm
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)

        # 因果自注意力
        self.attn = CausalSelfAttention(embed_dim, num_heads, dropout)

        # MLP
        self.mlp = MLP(embed_dim, dropout=dropout)

        # Dropout（用于残差连接）
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Args:
            x: [batch_size, seq_len, embed_dim]

        Returns:
            output: [batch_size, seq_len, embed_dim]
        """
        # 注意力子层（Pre-Norm + 残差连接）
        x = x + self.dropout(self.attn(self.ln1(x)))

        # MLP 子层（Pre-Norm + 残差连接）
        x = x + self.dropout(self.mlp(self.ln2(x)))

        return x


def demo_transformer_block():
    """演示 Transformer Block"""
    print("=" * 60)
    print("【1. Transformer Block 演示】")
    print("=" * 60)

    torch.manual_seed(42)

    # 参数
    batch_size = 2
    seq_len = 16
    embed_dim = 128
    num_heads = 4

    # 创建 Block
    block = TransformerBlock(embed_dim, num_heads)

    # 输入
    x = torch.randn(batch_size, seq_len, embed_dim)

    # 前向传播
    output = block(x)

    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")
    print(f"参数量: {sum(p.numel() for p in block.parameters()):,}")

    # 验证输出形状
    assert x.shape == output.shape, "形状不一致！"
    print("✓ 形状验证通过")


def demo_stacked_blocks():
    """演示堆叠多个 Block"""
    print("\n" + "=" * 60)
    print("【2. 堆叠多个 Block】")
    print("=" * 60)

    torch.manual_seed(42)

    # 参数
    batch_size = 1
    seq_len = 8
    embed_dim = 64
    num_heads = 4
    num_layers = 6  # 6 个 Block

    # 创建多个 Block
    blocks = nn.ModuleList([
        TransformerBlock(embed_dim, num_heads)
        for _ in range(num_layers)
    ])

    # 输入
    x = torch.randn(batch_size, seq_len, embed_dim)

    print(f"输入形状: {x.shape}")
    print(f"Block 数量: {num_layers}")
    print(f"\n逐层传播...")

    # 逐层传播
    for i, block in enumerate(blocks):
        x = block(x)
        print(f"  Block {i+1} 输出形状: {x.shape}")

    print(f"\n最终输出形状: {x.shape}")

    # 总参数量
    total_params = sum(p.numel() for p in blocks.parameters())
    print(f"总参数量: {total_params:,}")


def demo_gradient_flow():
    """演示梯度流动"""
    print("\n" + "=" * 60)
    print("【3. 梯度流动演示】")
    print("=" * 60)

    torch.manual_seed(42)

    # 简单设置
    batch_size = 1
    seq_len = 4
    embed_dim = 32
    num_heads = 2
    num_layers = 4

    # 创建模型
    blocks = nn.ModuleList([
        TransformerBlock(embed_dim, num_heads)
        for _ in range(num_layers)
    ])

    # 输入和目标
    x = torch.randn(batch_size, seq_len, embed_dim, requires_grad=True)
    target = torch.randn(batch_size, seq_len, embed_dim)

    # 前向传播
    output = x
    for block in blocks:
        output = block(output)

    # 计算损失
    loss = ((output - target) ** 2).mean()

    # 反向传播
    loss.backward()

    print(f"损失值: {loss.item():.4f}")
    print(f"\n梯度范数（检查梯度是否正常流动）:")
    print(f"  输入梯度范数: {x.grad.norm().item():.4f}")

    for i, block in enumerate(blocks):
        for name, param in block.named_parameters():
            if param.grad is not None and 'attn.q_proj.weight' in name:
                print(f"  Block {i+1} q_proj 梯度范数: {param.grad.norm().item():.4f}")

    print("""
如果梯度范数逐渐变小但仍为非零，说明梯度正常流动。
残差连接确保了梯度可以直接"跳过"各层。
    """)


def explain_gpt_sizes():
    """解释不同规模 GPT 的配置"""
    print("\n" + "=" * 60)
    print("【4. GPT 模型规模对比】")
    print("=" * 60)

    print("""
┌─────────┬────────┬─────────┬──────────┬────────────┐
│  模型   │ 层数   │ 注意力头│ 嵌入维度 │  参数量    │
├─────────┼────────┼─────────┼──────────┼────────────┤
│ GPT-2 S │   12   │    12   │   768    │   124M     │
│ GPT-2 M │   24   │    16   │  1024    │   355M     │
│ GPT-2 L │   36   │    20   │  1280    │   774M     │
│ GPT-2 XL│   48   │    25   │  1600    │  1.5B      │
├─────────┼────────┼─────────┼──────────┼────────────┤
│ GPT-3   │   96   │    96   │ 12288    │  175B      │
└─────────┴────────┴─────────┴──────────┴────────────┘

我们学习用的小模型配置：
- 层数: 6
- 注意力头: 6
- 嵌入维度: 384
- 参数量: ~10M

这足以在小数据集上训练，也能学到语言的基本规律。
    """)


def main():
    print("=" * 60)
    print("第9课: 完整的 Transformer Block")
    print("=" * 60)

    demo_transformer_block()
    demo_stacked_blocks()
    demo_gradient_flow()
    explain_gpt_sizes()

    print("\n" + "=" * 60)
    print("总结")
    print("=" * 60)
    print("""
Transformer Block 要点：

1. 结构：LayerNorm → 注意力 → 残差连接
         LayerNorm → MLP → 残差连接

2. Pre-Norm：归一化在子层之前（GPT 使用）

3. 残差连接：x + sublayer(x)，保证梯度流动

4. 多个 Block 堆叠形成深层网络

下一阶段：组装完整的 GPT 模型！
    """)


if __name__ == "__main__":
    main()
