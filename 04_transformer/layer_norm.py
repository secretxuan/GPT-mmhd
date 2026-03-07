"""
=================================================================================
第8课: Layer Normalization（层归一化）
=================================================================================

为什么需要归一化？
---------------
训练神经网络时，每层的输入分布会不断变化（内部协变量偏移）。
这会导致：
- 训练不稳定
- 收敛速度慢
- 需要更小的学习率

归一化的作用：让每层的输入分布更稳定。

BatchNorm vs LayerNorm
---------------------
两种主要的归一化方式：

BatchNorm：对"同一特征跨所有样本"归一化
LayerNorm：对"同一样本的所有特征"归一化

对于序列数据（如文本），LayerNorm 更合适：
- 每个样本独立处理
- 不受批量大小影响
- 处理变长序列更方便

GPT 使用的是 LayerNorm。

运行方式：python layer_norm.py
"""

import torch
import torch.nn as nn


class LayerNorm(nn.Module):
    """
    层归一化实现

    公式：
        output = (x - mean) / sqrt(var + eps) * gamma + beta

    其中：
        mean, var: 在最后一个维度上计算
        gamma, beta: 可学习的缩放和偏移参数
        eps: 防止除零的小常数
    """

    def __init__(self, embed_dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(embed_dim))  # 缩放参数
        self.beta = nn.Parameter(torch.zeros(embed_dim))  # 偏移参数

    def forward(self, x):
        """
        Args:
            x: [batch_size, seq_len, embed_dim]

        Returns:
            归一化后的张量，形状不变
        """
        # 在最后一个维度（embed_dim）上计算均值和方差
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)

        # 归一化
        x_norm = (x - mean) / torch.sqrt(var + self.eps)

        # 缩放和偏移
        output = self.gamma * x_norm + self.beta

        return output


def demo_layer_norm():
    """演示 LayerNorm 的效果"""
    print("=" * 60)
    print("【1. LayerNorm 演示】")
    print("=" * 60)

    torch.manual_seed(42)

    # 输入：2个样本，每个样本3个位置，每个位置4维特征
    x = torch.randn(2, 3, 4)

    print(f"输入形状: {x.shape}")
    print(f"输入数据:\n{x}\n")

    # 使用我们自己实现的 LayerNorm
    ln = LayerNorm(4)
    output = ln(x)

    print(f"归一化后:\n{output}\n")

    # 验证：每个位置的均值应接近0，标准差接近1
    print("验证归一化效果:")
    mean = output.mean(dim=-1)
    std = output.std(dim=-1)
    print(f"  均值（应接近0）: {mean}")
    print(f"  标准差（应接近1）: {std}")


def demo_pytorch_layernorm():
    """演示 PyTorch 内置的 LayerNorm"""
    print("\n" + "=" * 60)
    print("【2. PyTorch 内置 LayerNorm】")
    print("=" * 60)

    torch.manual_seed(42)
    x = torch.randn(2, 3, 4)

    # PyTorch 内置实现
    ln = nn.LayerNorm(4)
    output = ln(x)

    print(f"内置 LayerNorm 输出:\n{output}\n")
    print("""
实际使用时推荐用 PyTorch 内置的 nn.LayerNorm：
- 经过优化，速度更快
- 支持更多特性
- 与其他组件兼容性好
    """)


def explain_pre_norm():
    """解释 Pre-Norm vs Post-Norm"""
    print("\n" + "=" * 60)
    print("【3. Pre-Norm vs Post-Norm】")
    print("=" * 60)

    print("""
LayerNorm 放在哪里？

原始 Transformer (Post-Norm):
    输入 → 注意力 → Dropout → +残差 → LayerNorm → 输出
                    ↑___________|

GPT-2/3 (Pre-Norm):
    输入 → LayerNorm → 注意力 → Dropout → +残差 → 输出
              ↑                        |
              └────────────────────────┘

Pre-Norm 的优点：
1. 训练更稳定
2. 梯度流动更顺畅
3. 可以训练更深的网络

GPT-2 开始改用 Pre-Norm，这是现代 Transformer 的标准做法。
    """)


def demo_pre_norm_structure():
    """演示 Pre-Norm 结构"""
    print("\n" + "=" * 60)
    print("【4. Pre-Norm 结构代码】")
    print("=" * 60)

    print("""
# Pre-Norm Transformer Block 的结构
class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.attn = CausalSelfAttention(embed_dim, num_heads)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim)

    def forward(self, x):
        # 注意力子层（Pre-Norm）
        x = x + self.attn(self.ln1(x))  # 先归一化，再注意力，最后残差

        # MLP 子层（Pre-Norm）
        x = x + self.mlp(self.ln2(x))   # 先归一化，再MLP，最后残差

        return x

关键点：
1. LayerNorm 在子层之前
2. 残差连接跨越整个子层
3. 输入直接"跳连"到输出
    """)


def explain_rms_norm():
    """解释 RMSNorm（进阶）"""
    print("\n" + "=" * 60)
    print("【5. RMSNorm（进阶）】")
    print("=" * 60)

    print("""
现代大模型（如 LLaMA）使用 RMSNorm 替代 LayerNorm。

RMSNorm（Root Mean Square Normalization）：
    output = x / sqrt(mean(x²) + eps) * gamma

与 LayerNorm 的区别：
- LayerNorm: (x - mean) / std * gamma + beta
- RMSNorm: x / rms * gamma

RMSNorm 的优点：
1. 不计算均值，计算量更小
2. 没有 beta 参数
3. 实验效果相当或更好

Llama 2、Mistral 等模型都使用 RMSNorm。
    """)

    # 简单实现
    class RMSNorm(nn.Module):
        def __init__(self, embed_dim, eps=1e-5):
            super().__init__()
            self.eps = eps
            self.gamma = nn.Parameter(torch.ones(embed_dim))

        def forward(self, x):
            rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
            return self.gamma * x / rms

    torch.manual_seed(42)
    x = torch.randn(2, 4)
    rms_norm = RMSNorm(4)
    output = rms_norm(x)

    print(f"RMSNorm 示例:")
    print(f"  输入: {x[0].tolist()}")
    print(f"  输出: {output[0].tolist()}")


def main():
    print("=" * 60)
    print("第8课: Layer Normalization")
    print("=" * 60)

    demo_layer_norm()
    demo_pytorch_layernorm()
    explain_pre_norm()
    demo_pre_norm_structure()
    explain_rms_norm()

    print("\n" + "=" * 60)
    print("总结")
    print("=" * 60)
    print("""
LayerNorm 要点：

1. 让每个位置的特征归一化（均值0，标准差1）
2. GPT 使用 Pre-Norm（归一化在子层之前）
3. 有可学习的 gamma 和 beta 参数
4. 现代模型可能用 RMSNorm 替代

下一课：组装完整的 Transformer Block！
    """)


if __name__ == "__main__":
    main()
