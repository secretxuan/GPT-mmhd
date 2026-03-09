# GPT 从零实现

从零开始实现 GPT 模型的教学项目，涵盖从 PyTorch 基础到预训练和微调的完整流程。

## 项目结构

```
GPT-mmhd/
├── 01_basics/           # PyTorch 基础
│   ├── tensor_basics.py # Tensor 操作
│   ├── nn_basics.py     # 神经网络基础
│   └── backprop.py      # 反向传播
├── 02_tokenizer/        # 分词器
│   └── char_tokenizer.py
├── 03_attention/        # 注意力机制
│   ├── self_attention.py
│   └── causal_attention.py
├── 04_transformer/      # Transformer 组件
│   ├── layer_norm.py
│   ├── mlp.py
│   └── block.py
├── 05_gpt_model/        # GPT 模型
│   ├── model.py         # 完整 GPT 模型
│   └── generate.py      # 文本生成
├── 06_pretrain/         # 预训练
│   ├── config.py        # 训练配置
│   ├── dataset.py       # 数据集
│   └── train.py         # 训练脚本
├── 07_finetune/         # 微调
│   ├── sft.py           # 监督微调
│   └── lora.py          # LoRA 微调
├── data/                # 数据目录
│   └── download_*.py    # 数据下载脚本
├── checkpoints/         # 模型检查点
├── requirements.txt     # 依赖包
└── README.md
```

## 快速开始

### 1. 环境准备

```bash
# 创建虚拟环境
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows

# 安装依赖
pip install -r requirements.txt
```

### 2. 下载数据

```bash
cd data
python download_wikitext.py    # 下载 WikiText-103
python download_real_data.py   # 下载其他数据集
```

### 3. 训练模型

```bash
cd 06_pretrain
python train.py
```

### 4. 文本生成

```bash
cd 05_gpt_model
python generate.py
```

## 模型配置

项目提供多种预设配置，基于 [nanoGPT](https://github.com/karpathy/nanoGPT) 和 Chinchilla Scaling Laws：

| 配置 | 层数 | 头数 | 嵌入维度 | 序列长度 | 参数量 |
|------|------|------|----------|----------|--------|
| nano | 3 | 3 | 48 | 128 | ~10K |
| micro | 4 | 4 | 128 | 256 | ~0.4M |
| mini | 6 | 6 | 192 | 256 | ~1.3M |
| small | 6 | 6 | 384 | 512 | ~5M |
| medium | 8 | 8 | 512 | 512 | ~10M |

推荐使用 `mini` 配置训练 WikiText-103 数据集。

## 学习路线

1. **01_basics**: PyTorch 基础 - Tensor 操作、神经网络、反向传播
2. **02_tokenizer**: 字符级分词器
3. **03_attention**: 自注意力机制和因果注意力
4. **04_transformer**: Transformer 核心组件
5. **05_gpt_model**: 完整 GPT 模型组装
6. **06_pretrain**: 预训练流程
7. **07_finetune**: SFT 和 LoRA 微调

## 参考资料

- [nanoGPT](https://github.com/karpathy/nanoGPT) - Karpathy 的 GPT 实现
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Transformer 原始论文
- [Language Models are Unsupervised Multitask Learners](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) - GPT-2 论文
- [Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556) - Chinchilla 论文

## License

MIT License
