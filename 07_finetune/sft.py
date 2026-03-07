"""
=================================================================================
第13课: 监督微调 (Supervised Fine-Tuning, SFT)
=================================================================================

什么是微调？
----------
微调是在预训练模型的基础上，用特定任务的数据继续训练。

预训练 vs 微调：
- 预训练：在海量文本上学习"语言的规律"
- 微调：在特定任务数据上学习"如何完成任务"

SFT 的例子：
1. 预训练：模型学会了中文
2. 微调：教模型"按要求回答问题"

SFT 的数据格式：
    输入: "用户: 请介绍一下红楼梦\n助手: "
    目标: "用户: 请介绍一下红楼梦\n助手: 红楼梦是..."

运行方式：python sft.py
"""

import torch
import torch.nn as nn
import os
import sys

# 添加路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def explain_sft():
    """解释 SFT"""
    print("=" * 60)
    print("【1. 什么是 SFT？】")
    print("=" * 60)

    print("""
监督微调 (Supervised Fine-Tuning) 是 post-training 的第一步。

训练流程：
┌──────────────────────────────────────────────────────────────┐
│  预训练 (Pre-training)                                       │
│  数据：海量无标注文本（如整个互联网）                         │
│  目标：预测下一个 token                                       │
│  结果：模型学会了语言的基本规律                               │
└──────────────────────────────────────────────────────────────┘
                            ↓
┌──────────────────────────────────────────────────────────────┐
│  SFT (Supervised Fine-Tuning)                                │
│  数据：高质量的指令-回答对                                    │
│  目标：预测下一个 token（但现在是对话格式）                   │
│  结果：模型学会了如何"对话"                                   │
└──────────────────────────────────────────────────────────────┘
                            ↓
┌──────────────────────────────────────────────────────────────┐
│  RLHF (Reinforcement Learning from Human Feedback)           │
│  数据：人类偏好反馈                                           │
│  目标：最大化人类喜欢的回答                                   │
│  结果：模型输出更符合人类期望                                 │
└──────────────────────────────────────────────────────────────┘

ChatGPT 的训练流程就是：预训练 → SFT → RLHF
    """)


def explain_sft_data():
    """解释 SFT 数据格式"""
    print("\n" + "=" * 60)
    print("【2. SFT 数据格式】")
    print("=" * 60)

    print("""
SFT 需要高质量的"指令-回答"对数据。

格式示例：
┌────────────────────────────────────────────────────────────┐
│ {                                                          │
│   "instruction": "请用红楼梦的风格续写下面这句话",          │
│   "input": "林黛玉听了这话，",                              │
│   "output": "不觉心动，眼圈儿一红，便背过脸去..."           │
│ }                                                          │
└────────────────────────────────────────────────────────────┘

转换为训练数据：
    输入: "[INST] 请用红楼梦的风格续写下面这句话\\n林黛玉听了这话， [/INST]"
    目标: "不觉心动，眼圈儿一红，便背过脸去..."

关键点：
1. 使用特殊标记分隔用户输入和模型输出
2. 只在"输出"部分计算损失（输入部分的标签设为 -100）
3. 这样模型只学习生成回答，不学习重复问题
    """)


class SFTDataset(torch.utils.data.Dataset):
    """SFT 数据集"""

    def __init__(self, data, tokenizer, max_length=512):
        """
        Args:
            data: 指令数据列表，每项包含 'instruction', 'input', 'output'
            tokenizer: 分词器
            max_length: 最大长度
        """
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # 构建输入文本
        instruction = item.get('instruction', '')
        input_text = item.get('input', '')
        output = item.get('output', '')

        # 组合成对话格式
        prompt = f"用户: {instruction}\n{input_text}\n助手: "
        full_text = prompt + output

        # 编码
        prompt_ids = self.tokenizer.encode(prompt)
        full_ids = self.tokenizer.encode(full_text)

        # 截断
        if len(full_ids) > self.max_length:
            full_ids = full_ids[:self.max_length]

        # 创建标签（输入部分设为 -100，只计算输出的损失）
        labels = full_ids.copy()
        prompt_len = min(len(prompt_ids), len(full_ids))
        labels[:prompt_len] = [-100] * prompt_len

        # 创建输入（右移一位）
        input_ids = full_ids[:-1]
        labels = labels[1:]

        return torch.tensor(input_ids), torch.tensor(labels)


def demo_sft_training():
    """演示 SFT 训练"""
    print("\n" + "=" * 60)
    print("【3. SFT 训练演示】")
    print("=" * 60)

    # 模拟 SFT 数据
    sft_data = [
        {
            "instruction": "请用红楼梦的风格续写",
            "input": "贾宝玉",
            "output": "听了这话，不觉痴了，半晌方道：妹妹这是怎么说？"
        },
        {
            "instruction": "请用红楼梦的风格续写",
            "input": "林黛玉",
            "output": "冷笑两声道：你也不用哄我，我知道你的心。"
        },
    ]

    print("SFT 数据示例:")
    for i, item in enumerate(sft_data):
        print(f"\n样本 {i+1}:")
        print(f"  指令: {item['instruction']}")
        print(f"  输入: {item['input']}")
        print(f"  输出: {item['output']}")

    print("""
SFT 训练与预训练的区别：

1. 数据格式不同
   - 预训练：纯文本
   - SFT：指令-回答对

2. 损失计算不同
   - 预训练：所有位置都计算损失
   - SFT：只在输出部分计算损失

3. 训练规模不同
   - 预训练：海量数据，长训练时间
   - SFT：中等数据，较短训练时间
    """)


def explain_finetune_tips():
    """解释微调技巧"""
    print("\n" + "=" * 60)
    print("【4. 微调技巧】")
    print("=" * 60)

    print("""
微调时的关键技巧：

1. 学习率要小
   - 预训练学习率：~3e-4
   - 微调学习率：~1e-5 到 5e-5
   - 原因：模型已经训练好了，只需要小幅度调整

2. 数据质量很重要
   - 人工编写的高质量数据效果最好
   - 错误数据会"教坏"模型

3. 防止灾难性遗忘
   - 微调太久会忘记预训练学到的知识
   - 解决方案：
     a) 使用较小的学习率
     b) 混合预训练数据继续训练
     c) 使用参数高效微调（如 LoRA）

4. 多任务微调
   - 同时在多种任务数据上微调
   - 让模型保持多种能力
    """)


def explain_post_training():
    """解释 Post-training 完整流程"""
    print("\n" + "=" * 60)
    print("【5. Post-training 完整流程】")
    print("=" * 60)

    print("""
现代大模型的 Post-training 流程：

┌─────────────────────────────────────────────────────────────┐
│  Step 1: SFT (Supervised Fine-Tuning)                       │
│  - 在高质量对话数据上微调                                    │
│  - 让模型学会基本的对话能力                                  │
│  - 数据量：10K-100K 样本                                     │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  Step 2: Reward Model Training                              │
│  - 训练一个"打分模型"                                        │
│  - 输入：(问题, 回答)，输出：分数                            │
│  - 学习人类偏好                                              │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  Step 3: PPO (Proximal Policy Optimization)                 │
│  - 使用强化学习优化模型                                      │
│  - 目标：最大化 Reward Model 的打分                          │
│  - 同时保持与 SFT 模型的相似度                               │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  Step 4: DPO (Direct Preference Optimization)               │
│  - 直接优化人类偏好                                          │
│  - 不需要单独的 Reward Model                                 │
│  - 更简单、更稳定                                            │
└─────────────────────────────────────────────────────────────┘

ChatGPT、Claude、Llama 都采用类似的流程。
    """)


def main():
    print("=" * 60)
    print("第13课: 监督微调 (SFT)")
    print("=" * 60)

    explain_sft()
    explain_sft_data()
    demo_sft_training()
    explain_finetune_tips()
    explain_post_training()

    print("\n" + "=" * 60)
    print("总结")
    print("=" * 60)
    print("""
SFT 要点：

1. 在预训练基础上，用指令数据继续训练
2. 只在输出部分计算损失
3. 学习率要比预训练小
4. 数据质量很重要

下一课：LoRA - 高效微调技术
    """)


if __name__ == "__main__":
    main()
