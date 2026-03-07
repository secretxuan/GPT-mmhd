"""
=================================================================================
GPT 训练配置 - 基于 nanoGPT 和 Chinchilla Scaling Laws
=================================================================================

参考来源：
- nanoGPT: https://github.com/karpathy/nanoGPT
- minGPT: https://github.com/karpathy/minGPT
- Chinchilla: "Training Compute-Optimal Large Language Models"

Chinchilla 定律：训练 tokens ≈ 20 × 参数量

我们的数据 (wikitext-103):
- 518MB 文本，约 541M 字符
- 英文平均每 token 约 4 字符，所以约 135M tokens
- Chinchilla 最优参数量: 135M / 20 ≈ 7M

但为了训练速度，我们用更小的模型，接受欠训练。
"""

from dataclasses import dataclass


@dataclass
class GPTConfig:
    """GPT 模型配置"""
    vocab_size: int
    block_size: int = 256
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 192  # gpt-mini 配置
    dropout: float = 0.0  # 预训练用 0
    bias: bool = False    # nanoGPT 推荐 False


@dataclass
class TrainConfig:
    """训练配置"""
    # 数据
    batch_size: int = 64       # nanoGPT shakespeare 配置
    block_size: int = 256      # 上下文长度

    # 优化器 (nanoGPT 推荐值)
    learning_rate: float = 1e-3    # 小模型可用更高学习率
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95           # 比 0.999 更好
    grad_clip: float = 1.0

    # 学习率调度
    warmup_iters: int = 100
    lr_decay_iters: int = 5000
    min_lr: float = 1e-4           # 最终学习率 = learning_rate / 10

    # 训练
    max_iters: int = 5000
    eval_interval: int = 200
    eval_iters: int = 100
    checkpoint_interval: int = 1000


# ============================================================
# 预设配置 (基于 nanoGPT/minGPT 的成熟配置)
# ============================================================

# gpt-nano: 最小配置，快速验证
NANO_CONFIG = {
    'model': {'n_layer': 3, 'n_head': 3, 'n_embd': 48, 'block_size': 128},
    'train': {'batch_size': 64, 'learning_rate': 1e-3, 'max_iters': 2000},
    'params': '~10K',  # 参数量
    'usage': '快速验证代码'
}

# gpt-micro: 稍大一点
MICRO_CONFIG = {
    'model': {'n_layer': 4, 'n_head': 4, 'n_embd': 128, 'block_size': 256},
    'train': {'batch_size': 64, 'learning_rate': 1e-3, 'max_iters': 5000},
    'params': '~0.4M',
    'usage': '小数据集训练'
}

# gpt-mini: 推荐！适合 100-500MB 数据
MINI_CONFIG = {
    'model': {'n_layer': 6, 'n_head': 6, 'n_embd': 192, 'block_size': 256},
    'train': {'batch_size': 64, 'learning_rate': 1e-3, 'max_iters': 10000},
    'params': '~1.3M',
    'usage': 'wikitext-103 推荐'
}

# gpt-small: 接近 GPT-2 small 的缩小版
SMALL_CONFIG = {
    'model': {'n_layer': 6, 'n_head': 6, 'n_embd': 384, 'block_size': 512},
    'train': {'batch_size': 32, 'learning_rate': 3e-4, 'max_iters': 20000},
    'params': '~5M',
    'usage': '更大显存，更好效果'
}

# gpt-medium: 需要 8GB+ 显存
MEDIUM_CONFIG = {
    'model': {'n_layer': 8, 'n_head': 8, 'n_embd': 512, 'block_size': 512},
    'train': {'batch_size': 16, 'learning_rate': 3e-4, 'max_iters': 50000},
    'params': '~10M',
    'usage': '需要 GPU，效果好'
}


CONFIGS = {
    'nano': NANO_CONFIG,
    'micro': MICRO_CONFIG,
    'mini': MINI_CONFIG,    # 推荐
    'small': SMALL_CONFIG,
    'medium': MEDIUM_CONFIG,
}


def get_config(name='mini'):
    """获取配置"""
    if name not in CONFIGS:
        print(f"未知配置: {name}，可用: {list(CONFIGS.keys())}")
        name = 'mini'
    return CONFIGS[name]


def print_config_comparison():
    """打印配置对比"""
    print("""
================================================================================
配置对比 (基于 nanoGPT 和 Chinchilla Scaling Laws)
================================================================================

| 配置     | 层数 | 头数 | 嵌入维度 | 序列长度 | 参数量  | batch | 学习率  |
|----------|------|------|----------|----------|---------|-------|---------|
| nano     | 3    | 3    | 48       | 128      | ~10K    | 64    | 1e-3    |
| micro    | 4    | 4    | 128      | 256      | ~0.4M   | 64    | 1e-3    |
| mini ★   | 6    | 6    | 192      | 256      | ~1.3M   | 64    | 1e-3    |
| small    | 6    | 6    | 384      | 512      | ~5M     | 32    | 3e-4    |
| medium   | 8    | 8    | 512      | 512      | ~10M    | 16    | 3e-4    |

★ mini 配置推荐用于 wikitext-103 (518MB)

================================================================================
Chinchilla 最优训练 (20 tokens/参数)
================================================================================

| 数据量      | 最优参数量 | 推荐配置 |
|-------------|-----------|----------|
| 10MB (~2.5M tokens)   | ~125K    | nano  |
| 100MB (~25M tokens)   | ~1.25M   | mini  |
| 500MB (~125M tokens)  | ~6.25M   | small |
| 1GB (~250M tokens)    | ~12.5M   | medium|

我们的 wikitext-103 (518MB, ~135M tokens):
- Chinchilla 最优: ~7M 参数
- 我们用 mini (1.3M) 接受欠训练，但训练快

================================================================================
""")
    print()


if __name__ == "__main__":
    print_config_comparison()
