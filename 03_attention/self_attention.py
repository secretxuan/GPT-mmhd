"""
=================================================================================
第5课: 自注意力机制 (Self-Attention)
=================================================================================

自注意力是 Transformer 的核心创新，也是 GPT 的"灵魂"。

核心思想
-------
一句话概括："每个词都要和其他所有词'交流'，找出哪些词与自己最相关"

例子：
    "小明喜欢红色的苹果"

    当模型读到"苹果"时，它会问：
    - "小明"和"苹果"相关吗？相关性 = 0.1
    - "喜欢"和"苹果"相关吗？相关性 = 0.3
    - "红色"和"苹果"相关吗？相关性 = 0.9  ← 最相关！
    - "的"和"苹果"相关吗？相关性 = 0.2

    然后根据这些相关性，综合所有词的信息，得到"苹果"的表示。

Q、K、V 的含义
-------------
自注意力通过三个矩阵实现：
- Q (Query): 查询向量 - "我想找什么？"
- K (Key): 键向量 - "我是什么特征？"
- V (Value): 值向量 - "我的实际内容是什么？"

类比：图书馆找书
- Q = 你的搜索词（"Python入门"）
- K = 书的分类标签（书架上写的类别）
- V = 书的内容

计算过程：
    Attention(Q, K, V) = softmax(Q·K^T / √d) · V

运行方式：python self_attention.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def simple_attention_demo():
    """最简单的注意力演示"""
    print("=" * 60)
    print("【1. 直观理解：什么是注意力？】")
    print("=" * 60)

    # 假设有3个词的向量表示
    words = ["我", "爱", "你"]
    embeddings = torch.tensor([
        [1.0, 0.0, 0.0],  # "我"
        [0.0, 1.0, 0.0],  # "爱"
        [0.0, 0.0, 1.0],  # "你"
    ])

    print(f"词: {words}")
    print(f"向量表示:\n{embeddings}")
    print()

    # 注意力的核心：计算词与词之间的相关性
    # 相关性 = 向量的点积
    similarity = embeddings @ embeddings.T

    print("词与词之间的相关性（点积）:")
    print("        我      爱      你")
    for i, w in enumerate(words):
        print(f"  {w}   {similarity[i].tolist()}")
    print()

    # 归一化成概率（softmax）
    attention_weights = F.softmax(similarity, dim=1)

    print("归一化后的注意力权重（每行加起来=1）:")
    print("        我      爱      你")
    for i, w in enumerate(words):
        print(f"  {w}   {[f'{x:.2f}' for x in attention_weights[i].tolist()]}")
    print()

    # 用注意力权重加权求和
    output = attention_weights @ embeddings

    print("加权后的新表示:")
    print(output)
    print("""
解释：
- 每个词的新表示 = 所有序列词的加权平均
- 权重由词与词之间的相关性决定
- 相关性高的词贡献更大
    """)


class SimpleSelfAttention(nn.Module):
    """
    简化版自注意力实现
    帮助理解核心逻辑
    """

    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim

        # Q、K、V 的线性变换
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, x):
        """
        Args:
            x: 输入张量，形状 [batch, seq_len, embed_dim]

        Returns:
            输出张量，形状 [batch, seq_len, embed_dim]
        """
        batch_size, seq_len, embed_dim = x.shape

        # 1. 计算 Q、K、V
        Q = self.q_proj(x)  # [batch, seq_len, embed_dim]
        K = self.k_proj(x)  # [batch, seq_len, embed_dim]
        V = self.v_proj(x)  # [batch, seq_len, embed_dim]

        # 2. 计算注意力分数: Q·K^T
        # [batch, seq_len, embed_dim] @ [batch, embed_dim, seq_len]
        # = [batch, seq_len, seq_len]
        scores = Q @ K.transpose(-2, -1)

        # 3. 缩放（防止点积值过大）
        scores = scores / math.sqrt(embed_dim)

        # 4. softmax 归一化
        attention_weights = F.softmax(scores, dim=-1)

        # 5. 加权求和
        output = attention_weights @ V

        return output, attention_weights


def step_by_step_attention():
    """逐步演示自注意力的计算过程"""
    print("\n" + "=" * 60)
    print("【2. 逐步演示自注意力计算】")
    print("=" * 60)

    torch.manual_seed(42)

    # 参数设置
    batch_size = 1
    seq_len = 4  # 序列长度：4个词
    embed_dim = 8  # 嵌入维度

    # 模拟输入（已经过词嵌入层）
    x = torch.randn(batch_size, seq_len, embed_dim)

    print(f"输入形状: {x.shape} (batch={batch_size}, seq_len={seq_len}, embed_dim={embed_dim})")
    print(f"输入数据:\n{x[0]}\n")

    # 创建 Q、K、V 投影
    q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
    k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
    v_proj = nn.Linear(embed_dim, embed_dim, bias=False)

    # 步骤1：计算 Q、K、V
    print("步骤1: 计算 Q、K、V")
    Q = q_proj(x)
    K = k_proj(x)
    V = v_proj(x)
    print(f"  Q 形状: {Q.shape}")
    print(f"  K 形状: {K.shape}")
    print(f"  V 形状: {V.shape}\n")

    # 步骤2：计算注意力分数
    print("步骤2: 计算注意力分数 Q·K^T")
    scores = Q @ K.transpose(-2, -1)
    print(f"  分数矩阵形状: {scores.shape} (seq_len × seq_len)")
    print(f"  分数矩阵:\n{scores[0]}\n")

    # 步骤3：缩放
    print("步骤3: 缩放（除以 √d）")
    scores = scores / math.sqrt(embed_dim)
    print(f"  缩放因子: √{embed_dim} = {math.sqrt(embed_dim):.2f}")
    print(f"  缩放后:\n{scores[0]}\n")

    # 步骤4：softmax
    print("步骤4: Softmax 归一化（每行加起来=1）")
    attention_weights = F.softmax(scores, dim=-1)
    print(f"  注意力权重:\n{attention_weights[0]}")
    print(f"  第0行之和: {attention_weights[0, 0].sum():.4f}\n")

    # 步骤5：加权求和
    print("步骤5: 加权求和（注意力权重 × V）")
    output = attention_weights @ V
    print(f"  输出形状: {output.shape}")
    print(f"  输出:\n{output[0]}\n")

    print("""
总结自注意力计算流程：
1. 输入 x → 线性变换 → Q, K, V
2. Q·K^T → 注意力分数（相关性）
3. 缩放（除以√d）防止梯度消失
4. Softmax → 概率分布（权重）
5. 权重 × V → 加权输出
    """)


def multi_head_attention_intro():
    """多头注意力简介"""
    print("\n" + "=" * 60)
    print("【3. 多头注意力 (Multi-Head Attention)】")
    print("=" * 60)

    print("""
实际 GPT 使用的是"多头注意力"：

单头注意力：只用一组 Q、K、V
多头注意力：同时用多组 Q、K、V，然后合并结果

为什么用多头？
- 不同的"头"可以关注不同类型的关联
- 头1可能关注"主语-谓语"关系
- 头2可能关注"形容词-名词"关系
- 头3可能关注"代词-指代对象"关系

类比：
- 单头 = 只有一个"观察视角"
- 多头 = 有多个"专家"从不同角度分析

GPT-3 有 96 个注意力头！
    """)


class MultiHeadAttention(nn.Module):
    """
    多头自注意力实现

    这是 GPT 中实际使用的形式
    """

    def __init__(self, embed_dim, num_heads):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim 必须能被 num_heads 整除"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads  # 每个头的维度

        # Q、K、V 投影（一次性为所有头计算）
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        # 输出投影
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, x):
        batch_size, seq_len, _ = x.shape

        # 1. 计算 Q、K、V
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        # 2. 重塑为多头形式
        # [batch, seq, embed] → [batch, seq, heads, head_dim] → [batch, heads, seq, head_dim]
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # 3. 计算注意力
        scores = Q @ K.transpose(-2, -1) / math.sqrt(self.head_dim)
        attention_weights = F.softmax(scores, dim=-1)
        output = attention_weights @ V

        # 4. 合并多头
        # [batch, heads, seq, head_dim] → [batch, seq, embed]
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)

        # 5. 输出投影
        output = self.out_proj(output)

        return output


def demo_multi_head():
    """演示多头注意力"""
    print("\n" + "=" * 60)
    print("【4. 多头注意力实现演示】")
    print("=" * 60)

    torch.manual_seed(42)

    # 参数
    batch_size = 2
    seq_len = 8
    embed_dim = 64
    num_heads = 4  # 4个头，每个头维度=16

    mha = MultiHeadAttention(embed_dim, num_heads)
    x = torch.randn(batch_size, seq_len, embed_dim)

    output = mha(x)

    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")
    print(f"注意力头数: {num_heads}")
    print(f"每个头维度: {embed_dim // num_heads}")

    # 计算参数量
    total_params = sum(p.numel() for p in mha.parameters())
    print(f"参数量: {total_params:,}")


def main():
    print("=" * 60)
    print("第5课: 自注意力机制 (Self-Attention)")
    print("=" * 60)

    # 1. 直观理解
    simple_attention_demo()

    # 2. 逐步演示
    step_by_step_attention()

    # 3. 多头注意力介绍
    multi_head_attention_intro()

    # 4. 多头注意力实现
    demo_multi_head()

    print("\n" + "=" * 60)
    print("总结")
    print("=" * 60)
    print("""
自注意力 (Self-Attention) 要点：

1. 核心思想：每个词和其他所有词交流，找出相关性
2. Q、K、V：Query查询、Key键、Value值
3. 计算公式：Attention = softmax(Q·K^T / √d) · V
4. 多头注意力：多个头从不同角度分析

下一课：因果注意力（Causal Attention）—— GPT 为什么不能"看未来"
    """)


if __name__ == "__main__":
    main()
