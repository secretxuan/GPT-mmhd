"""
=================================================================================
第10课: 完整的 GPT 模型
=================================================================================

恭喜！现在我们要把所有学过的组件组装成完整的 GPT 模型。

GPT 模型的完整架构
-----------------

    输入文本: "红楼梦"
         ↓
    Tokenizer (分词器)
         ↓
    Token IDs: [1234, 5678, 9012]
         ↓
    ┌─────────────────────────────────────┐
    │  Token Embedding (词嵌入)           │
    │  Position Embedding (位置嵌入)      │
    │  Embedding Dropout                 │
    ├─────────────────────────────────────┤
    │  Transformer Block × N             │
    │  ┌─────────────────────────────┐   │
    │  │ LayerNorm → Attention → +  │   │
    │  │ LayerNorm → MLP → +        │   │
    │  └─────────────────────────────┘   │
    ├─────────────────────────────────────┤
    │  Final LayerNorm                   │
    │  Linear (语言模型头)               │
    └─────────────────────────────────────┘
         ↓
    Logits: [vocab_size] 的概率分布
         ↓
    预测下一个 token

运行方式：python model.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass


@dataclass
class GPTConfig:
    """GPT 模型配置"""
    vocab_size: int = 3000      # 词汇表大小（字符级：中文字符数）
    block_size: int = 256       # 最大序列长度
    n_layer: int = 6            # Transformer 层数
    n_head: int = 6             # 注意力头数
    n_embd: int = 384           # 嵌入维度
    dropout: float = 0.2        # Dropout 比率
    bias: bool = True           # 是否使用偏置


class CausalSelfAttention(nn.Module):
    """因果自注意力"""

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head

        # 合并的 QKV 投影（更高效）
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # 输出投影
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

        # Dropout
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        # 因果掩码（注册为 buffer，随模型移动）
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(config.block_size, config.block_size))
            .view(1, 1, config.block_size, config.block_size)
        )

    def forward(self, x):
        B, T, C = x.size()  # batch, sequence length, embedding dim

        # 计算 Q, K, V
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)

        # 重塑为多头形式
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # 注意力分数
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        # 加权求和
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # 输出投影
        y = self.resid_dropout(self.c_proj(y))
        return y


class MLP(nn.Module):
    """前馈神经网络"""

    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    """Transformer Block"""

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    """
    完整的 GPT 模型

    这是 GPT-2 的架构，也适用于 GPT-3
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        # 词嵌入和位置嵌入
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)  # token embedding
        self.wpe = nn.Embedding(config.block_size, config.n_embd)  # position embedding
        self.drop = nn.Dropout(config.dropout)

        # Transformer Blocks
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])

        # 最终的 LayerNorm
        self.ln_f = nn.LayerNorm(config.n_embd)

        # 语言模型头（预测下一个 token）
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # 权重共享：输入嵌入和输出层共享权重
        # 这样可以减少参数量，也有正则化效果
        self.wte.weight = self.lm_head.weight

        # 初始化权重
        self.apply(self._init_weights)

        # 打印模型参数量
        print(f"模型参数量: {self.get_num_params()/1e6:.2f}M")

    def _init_weights(self, module):
        """初始化权重"""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def get_num_params(self):
        """计算参数量"""
        return sum(p.numel() for p in self.parameters())

    def forward(self, idx, targets=None):
        """
        前向传播

        Args:
            idx: 输入 token IDs, [batch_size, seq_len]
            targets: 目标 token IDs, [batch_size, seq_len]（用于计算损失）

        Returns:
            logits: 预测的 logits, [batch_size, seq_len, vocab_size]
            loss: 如果提供了 targets，返回交叉熵损失
        """
        device = idx.device
        B, T = idx.size()

        assert T <= self.config.block_size, f"序列长度 {T} 超过最大长度 {self.config.block_size}"

        # 位置索引
        pos = torch.arange(0, T, dtype=torch.long, device=device)

        # 嵌入
        tok_emb = self.wte(idx)  # [B, T, n_embd]
        pos_emb = self.wpe(pos)  # [T, n_embd]
        x = self.drop(tok_emb + pos_emb)

        # Transformer Blocks
        for block in self.blocks:
            x = block(x)

        # 最终 LayerNorm
        x = self.ln_f(x)

        # 语言模型头
        logits = self.lm_head(x)  # [B, T, vocab_size]

        # 计算损失（如果提供了 targets）
        loss = None
        if targets is not None:
            # 交叉熵损失
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1
            )

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        生成文本

        Args:
            idx: 输入 token IDs, [batch_size, seq_len]
            max_new_tokens: 要生成的新 token 数量
            temperature: 采样温度（越高越随机）
            top_k: 只从概率最高的 k 个 token 中采样

        Returns:
            生成的 token IDs
        """
        for _ in range(max_new_tokens):
            # 如果序列太长，截断
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]

            # 前向传播
            logits, _ = self(idx_cond)

            # 只取最后一个位置的 logits
            logits = logits[:, -1, :] / temperature

            # 可选的 top-k 采样
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')

            # 转换为概率
            probs = F.softmax(logits, dim=-1)

            # 采样
            idx_next = torch.multinomial(probs, num_samples=1)

            # 拼接
            idx = torch.cat((idx, idx_next), dim=1)

        return idx


def demo_gpt_model():
    """演示 GPT 模型"""
    print("=" * 60)
    print("【1. 创建 GPT 模型】")
    print("=" * 60)

    # 配置
    config = GPTConfig(
        vocab_size=1000,  # 假设词汇表有 1000 个字符
        block_size=128,
        n_layer=4,
        n_head=4,
        n_embd=128,
        dropout=0.1
    )

    print(f"模型配置:")
    print(f"  词汇表大小: {config.vocab_size}")
    print(f"  最大序列长度: {config.block_size}")
    print(f"  层数: {config.n_layer}")
    print(f"  注意力头数: {config.n_head}")
    print(f"  嵌入维度: {config.n_embd}")

    # 创建模型
    model = GPT(config)

    # 测试前向传播
    batch_size = 2
    seq_len = 32
    idx = torch.randint(0, config.vocab_size, (batch_size, seq_len))

    logits, _ = model(idx)

    print(f"\n输入形状: {idx.shape}")
    print(f"输出形状: {logits.shape}")

    return model, config


def demo_generation():
    """演示文本生成"""
    print("\n" + "=" * 60)
    print("【2. 文本生成演示】")
    print("=" * 60)

    # 创建模型
    config = GPTConfig(vocab_size=100, block_size=64, n_layer=2, n_head=2, n_embd=64)
    model = GPT(config)

    # 随机输入（实际使用时应该是真实的 token IDs）
    idx = torch.randint(0, config.vocab_size, (1, 10))

    print(f"输入 token IDs: {idx[0].tolist()}")
    print("生成中...")

    # 生成
    output = model.generate(idx, max_new_tokens=20, temperature=1.0, top_k=10)

    print(f"输出 token IDs: {output[0].tolist()}")
    print(f"生成了 {output.size(1) - idx.size(1)} 个新 token")


def demo_training_step():
    """演示训练步骤"""
    print("\n" + "=" * 60)
    print("【3. 训练步骤演示】")
    print("=" * 60)

    # 创建模型
    config = GPTConfig(vocab_size=100, block_size=32, n_layer=2, n_head=2, n_embd=64)
    model = GPT(config)

    # 模拟数据
    batch_size = 4
    seq_len = 16

    # 输入和目标（目标是输入向后移一位）
    idx = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    targets = torch.randint(0, config.vocab_size, (batch_size, seq_len))

    # 前向传播
    logits, loss = model(idx, targets)

    print(f"输入形状: {idx.shape}")
    print(f"目标形状: {targets.shape}")
    print(f"Logits 形状: {logits.shape}")
    print(f"损失值: {loss.item():.4f}")

    # 反向传播
    loss.backward()

    # 检查梯度
    total_norm = 0
    for p in model.parameters():
        if p.grad is not None:
            total_norm += p.grad.norm().item() ** 2
    total_norm = total_norm ** 0.5

    print(f"梯度范数: {total_norm:.4f}")
    print("\n✓ 训练步骤完成！")


def explain_weight_tying():
    """解释权重共享"""
    print("\n" + "=" * 60)
    print("【4. 权重共享 (Weight Tying)】")
    print("=" * 60)

    print("""
GPT 使用"权重共享"技术：

    self.wte.weight = self.lm_head.weight

含义：
- 输入嵌入层 (wte) 和输出层 (lm_head) 共享相同的权重
- Token ID → 向量 和 向量 → Token ID 使用同一套参数

优点：
1. 减少参数量（节省内存）
2. 起到正则化作用（防止过拟合）
3. 实验证明效果更好

类比：
- 输入时把词转成向量（编码）
- 输出时把向量转回词（解码）
- 编码和解码用同一套"字典"更合理
    """)


def explain_config_tradeoffs():
    """解释配置权衡"""
    print("\n" + "=" * 60)
    print("【5. 模型配置权衡】")
    print("=" * 60)

    print("""
┌──────────────────────────────────────────────────────────────┐
│ 增大参数        │ 优点                   │ 缺点              │
├──────────────────────────────────────────────────────────────┤
│ vocab_size     │ 更好的分词             │ 更多参数          │
│ block_size     │ 更长的上下文           │ 更多显存          │
│ n_layer        │ 更深，更强             │ 更慢，更难训练    │
│ n_head         │ 更细的注意力           │ 计算量增加        │
│ n_embd         │ 更丰富的表示           │ 参数量平方增长    │
└──────────────────────────────────────────────────────────────┘

推荐配置（学习用）：
- vocab_size: 取决于数据（中文字符约3000-5000）
- block_size: 256（够用）
- n_layer: 6（平衡）
- n_head: 6（与层数相同）
- n_embd: 384（适中）
- 参数量: 约 10M

这能在普通电脑上训练，也能学到有意义的模式。
    """)


def main():
    print("=" * 60)
    print("第10课: 完整的 GPT 模型")
    print("=" * 60)

    demo_gpt_model()
    demo_generation()
    demo_training_step()
    explain_weight_tying()
    explain_config_tradeoffs()

    print("\n" + "=" * 60)
    print("总结")
    print("=" * 60)
    print("""
GPT 模型要点：

1. 结构：
   - Token Embedding + Position Embedding
   - N × Transformer Block
   - Final LayerNorm + Linear

2. 权重共享：输入嵌入和输出层共享权重

3. 训练目标：预测下一个 token（自回归）

4. 生成：逐步采样下一个 token

下一阶段：开始预训练！
    """)


if __name__ == "__main__":
    demo_gpt_model()
