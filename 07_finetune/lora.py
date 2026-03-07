"""
=================================================================================
第14课: LoRA - 低秩适应 (Low-Rank Adaptation)
=================================================================================

LoRA 是什么？
-----------
LoRA 是一种"参数高效微调"技术。

核心思想：
- 不直接训练原始大矩阵 W
- 而是训练两个小矩阵 A 和 B，其中 W' = W + BA
- A 和 B 的参数量远小于 W

原理：
    原始: Y = X @ W           W 的形状: [d, d], 参数量: d²

    LoRA: Y = X @ (W + BA)    B 的形状: [d, r]
                               A 的形状: [r, d]
                               参数量: 2*d*r

    当 r << d 时，参数量大幅减少！

例如：
    d = 4096, r = 8
    原始参数: 4096² = 16,777,216
    LoRA 参数: 2 * 4096 * 8 = 65,536
    减少了 256 倍！

LoRA 的优点：
1. 大幅减少显存占用
2. 训练更快
3. 可以为不同任务训练不同的 LoRA
4. 推理时可以无缝切换

运行方式：python lora.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class LoRALinear(nn.Module):
    """
    带 LoRA 的线性层

    公式: Y = X @ W + X @ A @ B
    其中 A 和 B 是低秩矩阵，r << d
    """

    def __init__(self, in_features, out_features, r=8, alpha=16, dropout=0.0):
        """
        Args:
            in_features: 输入维度
            out_features: 输出维度
            r: LoRA 秩（低秩矩阵的维度）
            alpha: 缩放因子
            dropout: Dropout 比率
        """
        super().__init__()

        # 原始线性层（冻结，不训练）
        self.linear = nn.Linear(in_features, out_features, bias=False)

        # LoRA 参数
        self.lora_A = nn.Parameter(torch.zeros(r, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))

        # 缩放因子
        self.scaling = alpha / r

        # Dropout
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # 初始化
        # A 用高斯初始化，B 初始化为 0
        # 这样初始时 LoRA 的输出为 0，不影响预训练模型
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        # 原始输出
        result = self.linear(x)

        # LoRA 增量
        lora_out = self.lora_dropout(x) @ self.lora_A.T @ self.lora_B.T
        result = result + lora_out * self.scaling

        return result

    def merge_weights(self):
        """合并 LoRA 权重到原始权重（用于推理）"""
        self.linear.weight.data += (self.lora_B @ self.lora_A) * self.scaling


def explain_lora_math():
    """解释 LoRA 的数学原理"""
    print("=" * 60)
    print("【1. LoRA 数学原理】")
    print("=" * 60)

    print("""
LoRA 的理论基础：低秩分解

假设权重更新 ΔW 可以分解为两个低秩矩阵的乘积：
    ΔW = B @ A

其中：
- B: [d_out, r]
- A: [r, d_in]
- r 是"秩"，通常 r << min(d_in, d_out)

为什么有效？
- 研究发现，模型微调时权重的变化通常是低秩的
- 即 ΔW 的"有效维度"远小于原始维度
- 所以用低秩矩阵来近似是合理的

参数量对比：
    原始: d_in × d_out
    LoRA: r × (d_in + d_out)

    当 r = 8, d = 4096:
    原始: 16,777,216
    LoRA: 65,536 (减少 256 倍)
    """)


def demo_lora_params():
    """演示 LoRA 参数量"""
    print("\n" + "=" * 60)
    print("【2. LoRA 参数量对比】")
    print("=" * 60)

    # 不同规模的对比
    configs = [
        (768, 768, 8),      # GPT-2 small
        (1024, 1024, 8),    # GPT-2 medium
        (4096, 4096, 8),    # Llama-7B
    ]

    print(f"{'维度 d':>10} | {'秩 r':>6} | {'原始参数':>12} | {'LoRA参数':>10} | {'比例':>8}")
    print("-" * 60)

    for d_in, d_out, r in configs:
        original = d_in * d_out
        lora_params = r * (d_in + d_out)
        ratio = original / lora_params

        print(f"{d_in:>10} | {r:>6} | {original:>12,} | {lora_params:>10,} | {ratio:>7.1f}x")

    print("""
可以看到，使用 LoRA 后参数量大幅减少！
这让我们可以在消费级 GPU 上微调大模型。
    """)


def demo_lora_layer():
    """演示 LoRA 层"""
    print("\n" + "=" * 60)
    print("【3. LoRA 层演示】")
    print("=" * 60)

    torch.manual_seed(42)

    # 参数
    in_features = 128
    out_features = 256
    r = 8

    # 创建 LoRA 线性层
    lora_linear = LoRALinear(in_features, out_features, r=r, alpha=16)

    # 输入
    x = torch.randn(2, 10, in_features)

    # 前向传播
    output = lora_linear(x)

    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")
    print(f"秩 r: {r}")
    print(f"缩放因子: {lora_linear.scaling}")

    # 参数量
    total_params = sum(p.numel() for p in lora_linear.parameters())
    lora_params = lora_linear.lora_A.numel() + lora_linear.lora_B.numel()

    print(f"\n参数统计:")
    print(f"  原始层参数: {lora_linear.linear.weight.numel():,}")
    print(f"  LoRA 参数: {lora_params:,}")
    print(f"  LoRA 占比: {lora_params/total_params*100:.1f}%")


def apply_lora_to_model():
    """演示如何对模型应用 LoRA"""
    print("\n" + "=" * 60)
    print("【4. 对 GPT 模型应用 LoRA】")
    print("=" * 60)

    print("""
对 GPT 模型应用 LoRA 的步骤：

1. 选择要应用 LoRA 的层
   - 通常是注意力层的 Q、K、V 投影
   - 也可以应用到所有线性层

2. 冻结原始参数
   for param in model.parameters():
       param.requires_grad = False

3. 替换为 LoRA 层
   for name, module in model.named_modules():
       if isinstance(module, nn.Linear) and should_apply_lora(name):
           lora_layer = LoRALinear(module.in_features, module.out_features)
           lora_layer.linear.weight.data = module.weight.data.clone()
           replace_module(model, name, lora_layer)

4. 训练（只有 LoRA 参数会更新）
   optimizer = Adam([p for p in model.parameters() if p.requires_grad], lr=1e-4)

典型配置：
- 秩 r: 8-64 (常用 8, 16, 32)
- alpha: 通常是 r 的 2 倍
- 目标层: q_proj, v_proj (效果最好)
    """)


def demo_lora_training():
    """演示 LoRA 训练"""
    print("\n" + "=" * 60)
    print("【5. LoRA 训练演示】")
    print("=" * 60)

    # 创建一个简单的模型
    class SimpleModel(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.fc1 = LoRALinear(dim, dim * 4, r=8)
            self.fc2 = LoRALinear(dim * 4, dim, r=8)

        def forward(self, x):
            x = F.gelu(self.fc1(x))
            x = self.fc2(x)
            return x

    dim = 64
    model = SimpleModel(dim)

    # 统计参数
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"模型参数统计:")
    print(f"  总参数: {total_params:,}")
    print(f"  可训练参数: {trainable_params:,}")
    print(f"  可训练占比: {trainable_params/total_params*100:.1f}%")

    # 模拟训练
    x = torch.randn(4, 10, dim)
    target = torch.randn(4, 10, dim)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    print("\n训练中...")
    for i in range(10):
        optimizer.zero_grad()
        output = model(x)
        loss = F.mse_loss(output, target)
        loss.backward()
        optimizer.step()

        if (i + 1) % 2 == 0:
            print(f"  Step {i+1}: loss = {loss.item():.4f}")

    print("\n✓ 训练完成")


def explain_lora_best_practices():
    """LoRA 最佳实践"""
    print("\n" + "=" * 60)
    print("【6. LoRA 最佳实践】")
    print("=" * 60)

    print("""
LoRA 配置建议：

1. 秩 (r) 的选择
   - r = 8: 大多数任务够用
   - r = 16-32: 复杂任务
   - r = 64+: 很少需要

2. alpha 的选择
   - 通常 alpha = 2 * r
   - 影响 LoRA 增量的权重

3. 应用位置
   - 只用 Q、V: 参数最少
   - Q、K、V、O: 效果更好
   - 全部线性层: 效果最好，参数最多

4. 学习率
   - LoRA 可以用较大的学习率
   - 通常 1e-4 到 5e-4

5. 合并权重
   - 训练后可以合并 LoRA 权重
   - 合并后推理无额外开销

常用 LoRA 库：
- Hugging Face PEFT
- Microsoft LoRA
- bitsandbytes (配合量化)
    """)


def main():
    print("=" * 60)
    print("第14课: LoRA - 低秩适应")
    print("=" * 60)

    explain_lora_math()
    demo_lora_params()
    demo_lora_layer()
    apply_lora_to_model()
    demo_lora_training()
    explain_lora_best_practices()

    print("\n" + "=" * 60)
    print("总结")
    print("=" * 60)
    print("""
LoRA 要点：

1. 用两个小矩阵 B、A 近似权重更新
2. 参数量大幅减少（几十到几百倍）
3. 只训练 LoRA 参数，冻结原始权重
4. 推理时可合并，无额外开销

恭喜！你已经学完了整个 GPT 训练流程！

回顾：
- Pre-training: 在海量数据上学习语言
- SFT: 学习对话格式
- LoRA: 高效微调

现在你可以：
1. 自己训练一个小型 GPT
2. 理解大模型的训练原理
3. 进行高效的模型微调
    """)


if __name__ == "__main__":
    main()
