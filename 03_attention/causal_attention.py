"""
=================================================================================
第6课: 因果注意力 (Causal Attention) - GPT 的核心
=================================================================================

问题：普通自注意力有什么问题？
---------------------------
普通自注意力中，每个词可以"看到"所有其他词（包括后面的词）。

但在文本生成中，这是"作弊"！
    输入: "今天天气很"
    预测: "好"

如果模型在预测时能看到后面的"好"，那还预测什么？

因果注意力的解决方案
-----------------
使用"掩码"（Mask）阻止模型看到未来的词：

    输入序列: [A, B, C, D]

    普通 Attention 权重矩阵 (每个词看所有词):
        A   B   C   D
    A [0.25, 0.25, 0.25, 0.25]  ← A 看到所有
    B [0.25, 0.25, 0.25, 0.25]  ← B 看到所有
    C [0.25, 0.25, 0.25, 0.25]  ← C 看到所有
    D [0.25, 0.25, 0.25, 0.25]  ← D 看到所有

    因果 Attention 权重矩阵 (只能看左边):
        A   B   C   D
    A [1.0, 0,   0,   0 ]  ← A 只能看自己
    B [0.5, 0.5, 0,   0 ]  ← B 能看 A、B
    C [0.33,0.33,0.33,0 ]  ← C 能看 A、B、C
    D [0.25,0.25,0.25,0.25] ← D 能看所有

这就是"因果"的含义：当前位置的输出只依赖于之前的位置。

运行方式：python causal_attention.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def visualize_causal_mask():
    """可视化因果掩码"""
    print("=" * 60)
    print("【1. 因果掩码可视化】")
    print("=" * 60)

    seq_len = 6

    # 创建下三角矩阵（包括对角线）
    mask = torch.tril(torch.ones(seq_len, seq_len))

    print(f"因果掩码 (1=可见, 0=不可见):")
    print("   位置0  位置1  位置2  位置3  位置4  位置5")
    for i in range(seq_len):
        row = [f"  {int(mask[i, j])}   " for j in range(seq_len)]
        print(f"位置{i}  " + "".join(row))

    print("""
解读：
- 位置0 只能看到自己 (0)
- 位置1 能看到 0, 1
- 位置2 能看到 0, 1, 2
- ... 以此类推

这就像人读书一样：读到第3个字时，你只知道前3个字，不知道后面的。
    """)


def demo_masked_softmax():
    """演示带掩码的 softmax"""
    print("\n" + "=" * 60)
    print("【2. 掩码 Softmax 实现】")
    print("=" * 60)

    # 模拟注意力分数
    scores = torch.tensor([
        [1.0, 2.0, 3.0, 4.0],
        [2.0, 1.0, 3.0, 0.0],
        [1.0, 2.0, 1.0, 2.0],
        [0.0, 1.0, 2.0, 3.0]
    ])

    print(f"原始注意力分数:\n{scores}\n")

    # 方法1：普通 softmax
    normal_attention = F.softmax(scores, dim=1)
    print(f"普通 Softmax:\n{normal_attention}")
    print(f"每行之和: {normal_attention.sum(dim=1).tolist()}\n")

    # 方法2：因果掩码 softmax
    # 把不可见位置设为 -inf，softmax 后会变成 0
    mask = torch.tril(torch.ones_like(scores))  # 下三角
    masked_scores = scores.masked_fill(mask == 0, float('-inf'))

    print(f"掩码后的分数 (-inf 表示不可见):\n{masked_scores}\n")

    causal_attention = F.softmax(masked_scores, dim=1)
    print(f"因果 Softmax:\n{causal_attention}")
    print(f"每行之和: {causal_attention.sum(dim=1).tolist()}")

    print("""
关键技巧：
- 把不可见位置设为 -inf
- exp(-inf) = 0
- 所以 softmax 后这些位置权重为 0
    """)


class CausalSelfAttention(nn.Module):
    """
    因果自注意力 - GPT 使用的注意力机制

    与普通自注意力的区别：
    1. 添加了因果掩码，只能看到左边
    2. 处理可变长度序列
    """

    def __init__(self, embed_dim, num_heads):
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

        # 因果掩码（注册为 buffer，不参与训练）
        # 注意：这里不预先创建，因为序列长度可能变化

    def forward(self, x):
        """
        Args:
            x: [batch_size, seq_len, embed_dim]

        Returns:
            output: [batch_size, seq_len, embed_dim]
        """
        batch_size, seq_len, _ = x.shape

        # 1. 计算 Q、K、V
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        # 2. 重塑为多头形式
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # 3. 计算注意力分数
        scores = Q @ K.transpose(-2, -1) / math.sqrt(self.head_dim)

        # 4. ★ 关键：应用因果掩码 ★
        # 创建下三角掩码，并应用到分数上
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device))
        scores = scores.masked_fill(causal_mask == 0, float('-inf'))

        # 5. Softmax
        attention_weights = F.softmax(scores, dim=-1)

        # 6. 加权求和
        output = attention_weights @ V

        # 7. 合并多头
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)

        # 8. 输出投影
        output = self.out_proj(output)

        return output, attention_weights


def demo_causal_attention():
    """演示因果注意力"""
    print("\n" + "=" * 60)
    print("【3. 因果注意力完整实现】")
    print("=" * 60)

    torch.manual_seed(42)

    # 参数
    batch_size = 1
    seq_len = 5
    embed_dim = 16
    num_heads = 2

    # 创建模型
    attention = CausalSelfAttention(embed_dim, num_heads)

    # 模拟输入
    x = torch.randn(batch_size, seq_len, embed_dim)

    # 前向传播
    output, weights = attention(x)

    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")
    print(f"注意力权重形状: {weights.shape}")  # [batch, heads, seq, seq]

    # 可视化第一个头的注意力权重
    print(f"\n第1个注意力头的权重矩阵:")
    attn = weights[0, 0]  # 取第一个batch，第一个head
    print("      位置0   位置1   位置2   位置3   位置4")
    for i in range(seq_len):
        row = [f"{attn[i, j]:.4f}" for j in range(seq_len)]
        print(f"位置{i}  " + "  ".join(row))

    print("""
注意：右上角全是0（因为 -inf → softmax → 0）
这证明模型确实"看不到"未来的信息！
    """)


def demo_why_causal():
    """解释为什么需要因果注意力"""
    print("\n" + "=" * 60)
    print("【4. 为什么 GPT 需要因果注意力？】")
    print("=" * 60)

    print("""
GPT 的任务是"预测下一个字符"，这是一个自回归任务。

例子：生成 "我爱北京天安门"

训练时：
    输入: "我"           预测: "爱"
    输入: "我爱"         预测: "北"
    输入: "我爱北"       预测: "京"
    输入: "我爱北京"     预测: "天"
    ...

推理（生成）时：
    步骤1: 输入 "我"              → 输出 "爱"
    步骤2: 输入 "我爱"            → 输出 "北"
    步骤3: 输入 "我爱北"          → 输出 "京"
    ...

关键点：
- 生成时，我们还没有后面的字！
- 如果训练时模型能"偷看"后面的字，推理时就无法正常工作
- 因果注意力确保训练和推理的一致性

这叫做"自回归"（Autoregressive）：
- 每个位置只能依赖之前的位置
- 一步步生成，像人写字一样
    """)


def compare_attention_types():
    """对比不同类型的注意力"""
    print("\n" + "=" * 60)
    print("【5. 注意力类型对比】")
    print("=" * 60)

    print("""
┌────────────────┬─────────────────────────────────────┐
│     类型       │              特点                    │
├────────────────┼─────────────────────────────────────┤
│ 自注意力       │ 每个位置看所有位置                   │
│ (Self-Attn)    │ 用于 BERT（理解任务）                │
├────────────────┼─────────────────────────────────────┤
│ 因果注意力     │ 每个位置只看之前的位置               │
│ (Causal Attn)  │ 用于 GPT（生成任务）                 │
├────────────────┼─────────────────────────────────────┤
│ 交叉注意力     │ Q来自一个序列，K/V来自另一个序列    │
│ (Cross-Attn)   │ 用于 Encoder-Decoder架构             │
└────────────────┴─────────────────────────────────────┘

GPT 只使用因果注意力！
BERT 只使用双向自注意力！
T5/Transformer 同时使用三种！
    """)


def demo_efficient_implementation():
    """高效实现技巧"""
    print("\n" + "=" * 60)
    print("【6. 高效实现技巧】")
    print("=" * 60)

    print("""
实际 GPT 实现中的优化：

1. Flash Attention (PyTorch 2.0+)
   - 自动优化，无需手动实现
   - 显著减少显存使用
   - 加速计算

   使用方式：
   with torch.backends.cuda.sdp_kernel(enable_flash=True):
       output = F.scaled_dot_product_attention(Q, K, V, is_causal=True)

2. KV Cache（推理优化）
   - 缓存之前计算的 K、V
   - 生成时不需要重新计算
   - 大幅加速推理

3. 分组查询注意力 (GQA)
   - GPT-4 / Llama 2 使用
   - 多个 Q 共享一组 K、V
   - 减少显存，加速推理

我们先用基础实现学习原理，后续可以优化。
    """)


def main():
    print("=" * 60)
    print("第6课: 因果注意力 (Causal Attention)")
    print("=" * 60)

    # 1. 掩码可视化
    visualize_causal_mask()

    # 2. 掩码 softmax
    demo_masked_softmax()

    # 3. 因果注意力实现
    demo_causal_attention()

    # 4. 为什么需要
    demo_why_causal()

    # 5. 类型对比
    compare_attention_types()

    # 6. 高效实现
    demo_efficient_implementation()

    print("\n" + "=" * 60)
    print("总结")
    print("=" * 60)
    print("""
因果注意力要点：

1. GPT 只能"从左到右"看，不能看未来
2. 实现方式：用 -inf 掩盖未来位置
3. 这是"自回归生成"的基础
4. 训练和推理保持一致

公式：
    Attention = softmax(Q·K^T + Mask) · V
                        ↑
                   因果掩码（上三角为 -inf）

下一阶段：组装完整的 Transformer Block！
    """)


if __name__ == "__main__":
    main()
