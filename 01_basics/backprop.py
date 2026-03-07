"""
=================================================================================
第3课: 反向传播 (Backpropagation)
=================================================================================

前两课我们学了 Tensor 和神经网络层，这节课学习神经网络如何"学习"。

核心问题：神经网络如何自动调整参数？
--------------------------------
答案：反向传播 + 梯度下降

简单理解：
1. 前向传播：输入数据，得到预测结果
2. 计算损失：预测结果和真实值的差距
3. 反向传播：计算"每个参数对损失的责任"（梯度）
4. 梯度下降：根据梯度调整参数，减小损失

类比：在山上（损失函数表面），想找到最低点（最小损失）
- 梯度 = 坡度方向
- 梯度下降 = 往下坡方向走一步

运行方式：python backprop.py
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

def demo_gradient_intuition():
    """演示梯度的直观理解"""
    print("\n" + "=" * 60)
    print("【1. 梯度的直观理解】")
    print("=" * 60)

    # 创建一个需要计算梯度的参数
    # requires_grad=True 表示要追踪这个变量的梯度
    w = torch.tensor([2.0], requires_grad=True)
    b = torch.tensor([0.0], requires_grad=True)

    # 定义函数：y = w * x + b
    x = torch.tensor([3.0])

    # 前向传播
    y = w * x + b
    print(f"x = {x.item()}")
    print(f"w = {w.item()}, b = {b.item()}")
    print(f"y = w * x + b = {y.item()}")

    # 假设我们希望 y = 10
    target = torch.tensor([10.0])

    # 计算损失（均方误差）
    loss = (y - target) ** 2
    print(f"\n目标值: {target.item()}")
    print(f"预测值: {y.item()}")
    print(f"损失: {loss.item()}")

    # 反向传播：计算梯度
    loss.backward()

    # 查看梯度
    print(f"\n反向传播后:")
    print(f"  w.grad = {w.grad.item()}")  # d(loss)/d(w) = 2*(y-target)*x
    print(f"  b.grad = {b.grad.item()}")  # d(loss)/d(b) = 2*(y-target)

    # 梯度的含义：
    # w.grad > 0 表示：w 增大 → loss 增大 → 应该减小 w
    # w.grad < 0 表示：w 增大 → loss 减小 → 应该增大 w

    print("""
梯度解读：
- w.grad = 12 表示：w 每增加 1，loss 大约增加 12
- 所以我们应该减小 w 来降低 loss
    """)


def demo_gradient_descent():
    """演示梯度下降过程"""
    print("\n" + "=" * 60)
    print("【2. 梯度下降过程】")
    print("=" * 60)

    # 初始化参数
    w = torch.tensor([2.0], requires_grad=True)
    b = torch.tensor([0.0], requires_grad=True)

    x = torch.tensor([3.0])
    target = torch.tensor([10.0])
    learning_rate = 0.1  # 学习率：每步走多远

    print(f"初始值: w = {w.item():.2f}, b = {b.item():.2f}")
    print(f"目标: y = {target.item():.2f} (当 x = {x.item():.2f})")
    print(f"学习率: {learning_rate}")
    print("\n迭代过程:")

    for step in range(10):
        # 1. 前向传播
        y = w * x + b
        loss = (y - target) ** 2

        # 2. 反向传播
        loss.backward()

        # 3. 梯度下降（手动更新参数）
        with torch.no_grad():  # 更新时不需要计算梯度
            w -= learning_rate * w.grad
            b -= learning_rate * b.grad

        # 4. 清空梯度（重要！否则会累积）
        w.grad.zero_()
        b.grad.zero_()

        print(f"Step {step+1}: w = {w.item():.4f}, b = {b.item():.4f}, loss = {loss.item():.4f}")

    print(f"\n最终结果: y = {w.item():.4f} * x + {b.item():.4f}")
    print(f"理想结果: y = 3.0 * x + 1.0 (因为 3*3+1=10)")


def demo_optimizer():
    """演示使用 PyTorch 优化器"""
    print("\n" + "=" * 60)
    print("【3. 使用 PyTorch 优化器】")
    print("=" * 60)

    # 实际训练中，我们不会手动更新参数
    # 而是使用 optimizer 来自动完成

    # 定义模型和参数
    model = nn.Linear(1, 1)

    # 创建优化器（SGD = 随机梯度下降）
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    # 创建优化器（Adam = 更智能的优化器，GPT 训练常用）
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    print(f"优化器: {optimizer.__class__.__name__}")
    print(f"学习率: {optimizer.defaults['lr']}")
    print(f"管理 {len(list(model.parameters()))} 组参数")

    # 训练数据
    X = torch.linspace(-5, 5, 50).unsqueeze(1)
    y = 2 * X + 1 + torch.randn_like(X) * 0.5

    # 损失函数
    criterion = nn.MSELoss()

    print(f"\n训练数据: {X.shape[0]} 个样本")
    print("目标: 学习 y = 2x + 1")

    # 训练循环（简化版）
    losses = []
    for epoch in range(50):
        # 标准训练循环的 5 个步骤：

        # 1. 清空梯度
        optimizer.zero_grad()

        # 2. 前向传播
        predictions = model(X)

        # 3. 计算损失
        loss = criterion(predictions, y)

        # 4. 反向传播（计算梯度）
        loss.backward()

        # 5. 更新参数（梯度下降）
        optimizer.step()

        losses.append(loss.item())

    print(f"\n训练完成!")
    print(f"  学习到的权重: {model.weight.item():.4f} (目标: 2.0)")
    print(f"  学习到的偏置: {model.bias.item():.4f} (目标: 1.0)")
    print(f"  最终损失: {losses[-1]:.4f}")

    return losses


def demo_why_zero_grad():
    """演示为什么要 zero_grad()"""
    print("\n" + "=" * 60)
    print("【4. 为什么需要 zero_grad()】")
    print("=" * 60)

    w = torch.tensor([1.0], requires_grad=True)

    # 第一次反向传播
    y1 = w ** 2
    y1.backward()
    print(f"第一次 backward() 后, w.grad = {w.grad.item()}")  # 2

    # 如果不清零，第二次会累加
    y2 = w ** 2
    y2.backward()
    print(f"第二次 backward() 后, w.grad = {w.grad.item()}")  # 4 (2+2)

    # 正确做法：每次迭代前清零
    w.grad.zero_()
    y3 = w ** 2
    y3.backward()
    print(f"清零后再 backward(), w.grad = {w.grad.item()}")  # 2

    print("""
结论：
- PyTorch 默认会累积梯度
- 每次迭代前必须调用 optimizer.zero_grad() 或 w.grad.zero_()
- 否则梯度会越来越大，训练会出问题
    """)


def demo_no_grad():
    """演示 torch.no_grad() 的用途"""
    print("\n" + "=" * 60)
    print("【5. torch.no_grad() 的用途】")
    print("=" * 60)

    model = nn.Linear(2, 1)

    # 正常模式：会计算梯度
    x = torch.randn(1, 2)
    y = model(x)
    print(f"正常模式: y.requires_grad = {y.requires_grad}")

    # no_grad 模式：不计算梯度
    with torch.no_grad():
        y = model(x)
        print(f"no_grad 模式: y.requires_grad = {y.requires_grad}")

    print("""
使用场景：
1. 模型推理时（不需要梯度，节省内存）
2. 手动更新参数时
3. 评估/测试时

在 GPT 生成文本时，我们使用 no_grad 模式，因为只是推理。
    """)


def demo_training_loop_structure():
    """演示完整训练循环结构"""
    print("\n" + "=" * 60)
    print("【6. 标准训练循环结构】")
    print("=" * 60)

    print("""
# ============================================
# GPT 训练循环的标准结构（伪代码）
# ============================================

model = GPT()
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    for batch in dataloader:
        # 1. 获取数据
        input_ids = batch['input_ids']
        targets = batch['targets']

        # 2. 清空梯度
        optimizer.zero_grad()

        # 3. 前向传播
        logits = model(input_ids)

        # 4. 计算损失
        loss = F.cross_entropy(logits, targets)

        # 5. 反向传播
        loss.backward()

        # 6. 梯度裁剪（防止梯度爆炸）
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # 7. 更新参数
        optimizer.step()

        # 8. 记录/打印
        print(f"loss: {loss.item():.4f}")

# ============================================
关键点：
- zero_grad() 在 backward() 之前
- backward() 计算梯度，但不更新参数
- optimizer.step() 才真正更新参数
- clip_grad_norm_ 防止梯度过大导致训练不稳定
    """)


def main():
    print("=" * 60)
    print("第3课: 反向传播 (Backpropagation)")
    print("=" * 60)

    # 1. 梯度的直观理解
    demo_gradient_intuition()

    # 2. 梯度下降过程
    demo_gradient_descent()

    # 3. 使用优化器
    demo_optimizer()

    # 4. 为什么需要 zero_grad
    demo_why_zero_grad()

    # 5. no_grad 的用途
    demo_no_grad()

    # 6. 训练循环结构
    demo_training_loop_structure()

    print("\n" + "=" * 60)
    print("恭喜！你已经掌握了 PyTorch 训练的核心概念")
    print("=" * 60)
    print("""
回顾：
1. Tensor - 数据容器
2. nn.Linear, nn.Module - 网络结构
3. loss.backward() - 计算梯度
4. optimizer.step() - 更新参数
5. optimizer.zero_grad() - 清空梯度

下一阶段：我们开始实现 GPT 的核心组件 - 分词器！
    """)


if __name__ == "__main__":
    main()
