"""
=================================================================================
第7课: MLP（前馈神经网络）
=================================================================================

Transformer Block 的结构
----------------------
每个 Transformer Block 包含两个子层：
1. 注意力层（我们已经学过）
2. MLP 层（这节课的内容）

结构图：
    输入
      │
      ├──────────────────┐
      ↓                  │
    LayerNorm            │
      ↓                  │
    注意力  ←──────┘ 残差连接
      │
      ├──────────────────┐
      ↓                  │
    LayerNorm            │
      ↓                  │
    MLP  ←──────┘ 残差连接
      │
    输出

MLP 的作用
---------
注意力层负责"词与词之间的交流"
MLP 层负责"每个词内部的处理"

MLP 结构（GPT 使用）：
    x → Linear → GELU → Linear → 输出

中间维度会扩大 4 倍：
    embed_dim → 4*embed_dim → embed_dim

运行方式：python mlp.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    """
    GPT 使用的前馈神经网络

    结构：Linear → GELU → Linear

    特点：
    1. 中间维度扩大 4 倍
    2. 使用 GELU 激活函数（比 ReLU 更平滑）
    3. 两个线性层之间没有偏置（某些实现）
    """

    def __init__(self, embed_dim, hidden_dim=None, dropout=0.0):
        super().__init__()
        hidden_dim = hidden_dim or 4 * embed_dim  # 默认扩大4倍

        self.fc1 = nn.Linear(embed_dim, hidden_dim)  # 扩展
        self.fc2 = nn.Linear(hidden_dim, embed_dim)  # 压缩
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Args:
            x: [batch_size, seq_len, embed_dim]

        Returns:
            output: [batch_size, seq_len, embed_dim]
        """
        # 扩展 → 激活 → 压缩 → Dropout
        x = self.fc1(x)        # [batch, seq, hidden_dim]
        x = F.gelu(x)          # GELU 激活
        x = self.fc2(x)        # [batch, seq, embed_dim]
        x = self.dropout(x)    # 防止过拟合
        return x


def demo_mlp():
    """演示 MLP 的作用"""
    print("=" * 60)
    print("【1. MLP 结构演示】")
    print("=" * 60)

    torch.manual_seed(42)

    # 参数
    batch_size = 2
    seq_len = 8
    embed_dim = 64

    # 创建 MLP
    mlp = MLP(embed_dim)

    # 输入
    x = torch.randn(batch_size, seq_len, embed_dim)

    # 前向传播
    output = mlp(x)

    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")
    print(f"中间层维度: {4 * embed_dim}")

    # 查看参数
    print(f"\nMLP 参数:")
    print(f"  fc1: {embed_dim} → {4 * embed_dim}")
    print(f"  fc2: {4 * embed_dim} → {embed_dim}")
    print(f"  总参数量: {sum(p.numel() for p in mlp.parameters()):,}")


def explain_gelu():
    """解释 GELU 激活函数"""
    print("\n" + "=" * 60)
    print("【2. GELU vs ReLU】")
    print("=" * 60)

    print("""
ReLU (Rectified Linear Unit):
    ReLU(x) = max(0, x)

    优点：简单，计算快
    缺点：负数部分完全为0，可能丢失信息

GELU (Gaussian Error Linear Unit):
    GELU(x) = x * Φ(x)
    其中 Φ(x) 是标准正态分布的累积分布函数

    优点：
    - 更平滑（可导处处连续）
    - 负数部分不完全为0
    - 在 Transformer 中效果更好

GPT、BERT、现代大模型都使用 GELU！
    """)

    # 可视化对比
    x = torch.linspace(-3, 3, 100)
    relu_out = F.relu(x)
    gelu_out = F.gelu(x)

    print(f"\n对比 x = -2 时:")
    print(f"  ReLU(-2) = {F.relu(torch.tensor(-2.0)).item():.4f}")
    print(f"  GELU(-2) = {F.gelu(torch.tensor(-2.0)).item():.4f}")

    print(f"\n对比 x = 2 时:")
    print(f"  ReLU(2) = {F.relu(torch.tensor(2.0)).item():.4f}")
    print(f"  GELU(2) = {F.gelu(torch.tensor(2.0)).item():.4f}")


def explain_why_expand():
    """解释为什么要先扩展再压缩"""
    print("\n" + "=" * 60)
    print("【3. 为什么中间层要扩大 4 倍？】")
    print("=" * 60)

    print("""
MLP 的结构：embed_dim → 4*embed_dim → embed_dim

为什么要先扩大再压缩？

1. 增加表达能力
   - 更大的中间空间可以学习更复杂的特征组合
   - 类似"展开"思考，再"总结"

2. 分离特征学习
   - 第一层：学习特征的组合
   - 激活函数：引入非线性
   - 第二层：将组合特征映射回原始空间

3. 实验经验
   - 4倍是实验得出的"甜蜜点"
   - 太小：表达能力不足
   - 太大：参数过多，容易过拟合

类比：
- 注意力层：人们互相交流
- MLP：每个人独立思考（在更大的思维空间中）
    """)


def demo_position_wise():
    """演示 MLP 是逐位置处理的"""
    print("\n" + "=" * 60)
    print("【4. MLP 是'位置独立'的】")
    print("=" * 60)

    print("""
MLP 的一个重要特点：每个位置独立处理

输入形状: [batch, seq_len, embed_dim]

处理方式：
- 位置 0 的向量 → MLP → 位置 0 的新向量
- 位置 1 的向量 → MLP → 位置 1 的新向量
- ...
- 位置 N 的向量 → MLP → 位置 N 的新向量

不同位置之间没有交互！

这与注意力层不同：
- 注意力层：位置之间会交互（计算相关性）
- MLP 层：每个位置独立处理

组合起来的效果：
- 注意力层：收集上下文信息
- MLP 层：处理和整合这些信息
    """)

    # 验证位置独立性
    torch.manual_seed(42)
    embed_dim = 8
    mlp = MLP(embed_dim)

    # 创建两个输入
    x1 = torch.randn(1, 1, embed_dim)  # 只有位置0
    x2 = torch.randn(1, 2, embed_dim)  # 位置0和位置1
    x2[:, 0, :] = x1[:, 0, :]  # 位置0相同

    output1 = mlp(x1)
    output2 = mlp(x2)

    # 验证位置0的输出是否相同
    print(f"\n验证位置独立性:")
    print(f"  单独处理位置0: {output1[0, 0, :3].tolist()}")
    print(f"  位置0在有其他位置时: {output2[0, 0, :3].tolist()}")
    print(f"  是否相同: {torch.allclose(output1[0, 0], output2[0, 0])}")


def main():
    print("=" * 60)
    print("第7课: MLP（前馈神经网络）")
    print("=" * 60)

    demo_mlp()
    explain_gelu()
    explain_why_expand()
    demo_position_wise()

    print("\n" + "=" * 60)
    print("总结")
    print("=" * 60)
    print("""
MLP 要点：

1. 结构：Linear → GELU → Linear
2. 中间层扩大 4 倍
3. 逐位置独立处理（不改变序列长度）
4. 与注意力层配合使用

下一课：LayerNorm - 训练稳定的关键
    """)


if __name__ == "__main__":
    main()
