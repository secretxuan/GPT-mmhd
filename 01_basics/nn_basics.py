"""
=================================================================================
第2课: 神经网络基础
=================================================================================

上一课我们学了 Tensor，这节课学习如何用 Tensor 构建神经网络。

什么是神经网络？
-------------
神经网络就是一个"函数"，输入数据，输出预测结果。

    输入 x → [神经网络] → 输出 y

最简单的神经网络就是"线性层"（也叫全连接层）：
    y = W·x + b

其中：
    W = 权重矩阵 (可学习的参数)
    b = 偏置向量 (可学习的参数)
    · = 矩阵乘法

神经网络"学习"的过程就是不断调整 W 和 b 的值，
让输出 y 越来越接近我们想要的答案。

运行方式：python nn_basics.py
"""

import torch
import torch.nn as nn

def main():
    print("=" * 60)
    print("第2课: 神经网络基础")
    print("=" * 60)

    # -------------------------------------------------------------------------
    # 1. 线性层 (nn.Linear) - 神经网络的基本单元
    # -------------------------------------------------------------------------
    print("\n【1. 线性层 (nn.Linear)】")
    print("-" * 40)

    # nn.Linear(in_features, out_features)
    # in_features: 输入维度
    # out_features: 输出维度

    # 创建一个线性层：输入4维，输出2维
    linear = nn.Linear(4, 2)

    print(f"线性层: {linear}")
    print(f"权重 W 的形状: {linear.weight.shape}")  # 2×4
    print(f"偏置 b 的形状: {linear.bias.shape}")    # 2

    # 使用这个线性层
    x = torch.randn(3, 4)  # 批量大小为3，每个样本4维
    print(f"\n输入 x 的形状: {x.shape}")

    y = linear(x)
    print(f"输出 y 的形状: {y.shape}")  # 3×2

    # 手动验证：y = x @ W.T + b
    y_manual = x @ linear.weight.T + linear.bias
    print(f"手动计算结果是否一致: {torch.allclose(y, y_manual)}")

    # -------------------------------------------------------------------------
    # 2. 激活函数 - 让神经网络能学习非线性关系
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("【2. 激活函数】")
    print("=" * 60)

    # 如果只有线性层，无论叠加多少层，最终还是一个线性函数
    # 激活函数引入"非线性"，让神经网络能拟合任意复杂的函数

    x = torch.linspace(-3, 3, 100)  # 从-3到3的100个点

    # ReLU: 最常用的激活函数
    # ReLU(x) = max(0, x)  负数变0，正数不变
    relu = nn.ReLU()
    y_relu = relu(x)
    print(f"ReLU(-2) = {relu(torch.tensor(-2.0))}")  # 0
    print(f"ReLU(2) = {relu(torch.tensor(2.0))}")    # 2

    # GELU: GPT 使用的激活函数（更平滑）
    gelu = nn.GELU()
    y_gelu = gelu(x)
    print(f"\nGELU(-2) ≈ {gelu(torch.tensor(-2.0)):.4f}")  # 接近0但不为0
    print(f"GELU(2) ≈ {gelu(torch.tensor(2.0)):.4f}")     # 接近2

    # Softmax: 把输出变成概率分布（所有值加起来等于1）
    # 常用于多分类问题的最后一层
    logits = torch.tensor([2.0, 1.0, 0.1])  # 原始分数
    probs = torch.softmax(logits, dim=0)
    print(f"\nSoftmax 示例:")
    print(f"  原始分数: {logits}")
    print(f"  概率分布: {probs}")
    print(f"  概率之和: {probs.sum():.6f}")  # 等于1

    # -------------------------------------------------------------------------
    # 3. 构建神经网络模块 (nn.Module)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("【3. 自定义神经网络模块】")
    print("=" * 60)

    class SimpleNet(nn.Module):
        """
        一个简单的神经网络：
        输入 → 线性层 → ReLU → 线性层 → 输出

        这是经典的"多层感知机"(MLP)结构
        """

        def __init__(self, input_dim, hidden_dim, output_dim):
            super().__init__()
            # 定义网络层
            self.fc1 = nn.Linear(input_dim, hidden_dim)  # 第一层
            self.relu = nn.ReLU()                        # 激活函数
            self.fc2 = nn.Linear(hidden_dim, output_dim) # 第二层

        def forward(self, x):
            """
            前向传播：定义数据如何流过网络

            x → fc1 → relu → fc2 → 输出
            """
            x = self.fc1(x)
            x = self.relu(x)
            x = self.fc2(x)
            return x

    # 创建网络实例
    net = SimpleNet(input_dim=10, hidden_dim=20, output_dim=3)
    print(f"网络结构:\n{net}")

    # 查看网络参数
    print(f"\n网络参数总数: {sum(p.numel() for p in net.parameters())}")

    # 使用网络
    x = torch.randn(5, 10)  # 批量5个样本，每个10维
    output = net(x)
    print(f"\n输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")  # 5×3

    # -------------------------------------------------------------------------
    # 4. 损失函数 - 衡量预测与真实值的差距
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("【4. 损失函数】")
    print("=" * 60)

    # 交叉熵损失 (CrossEntropyLoss) - 分类任务最常用
    # 用于预测"下一个字符是什么"这种多分类问题
    criterion = nn.CrossEntropyLoss()

    # 假设我们有3个类别（如字符A、B、C）
    # 模型输出（未归一化的分数）
    predictions = torch.tensor([[2.0, 1.0, 0.1]])  # 批量1，3类

    # 真实标签（第0类，即字符A）
    targets = torch.tensor([0])

    loss = criterion(predictions, targets)
    print(f"预测分数: {predictions}")
    print(f"真实标签: {targets}")
    print(f"交叉熵损失: {loss.item():.4f}")

    # 损失越小，说明预测越准确
    # 如果预测完全正确（第0类分数很高），损失接近0

    # -------------------------------------------------------------------------
    # 5. 完整示例：训练一个小模型
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("【5. 完整示例：训练一个小模型】")
    print("=" * 60)

    # 任务：学习一个简单的映射 y = 2x + 1
    # 这是一个回归问题，使用 MSELoss

    # 创建数据
    X_train = torch.linspace(-5, 5, 100).unsqueeze(1)  # 100个样本，1维
    y_train = 2 * X_train + 1  # 真实关系

    # 添加一些噪声
    y_train += torch.randn_like(y_train) * 0.5

    # 创建简单模型
    model = nn.Linear(1, 1)  # 输入1维，输出1维

    # 定义损失函数和优化器
    criterion = nn.MSELoss()  # 均方误差
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    print(f"训练前: weight = {model.weight.item():.4f}, bias = {model.bias.item():.4f}")
    print("目标: weight = 2.0, bias = 1.0")
    print("\n开始训练...")

    # 训练循环
    for epoch in range(100):
        # 1. 前向传播
        predictions = model(X_train)
        loss = criterion(predictions, y_train)

        # 2. 反向传播（计算梯度）
        optimizer.zero_grad()  # 清空之前的梯度
        loss.backward()        # 计算新梯度

        # 3. 更新参数
        optimizer.step()

        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1:3d}: loss = {loss.item():.4f}")

    print(f"\n训练后: weight = {model.weight.item():.4f}, bias = {model.bias.item():.4f}")
    print("已经学习到接近目标值！")

    # -------------------------------------------------------------------------
    # 6. GPT 中的关键层
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("【6. GPT 中的关键层】")
    print("=" * 60)

    print("""
GPT 模型主要使用以下层：

1. nn.Linear - 线性变换（Q、K、V投影，MLP）
2. nn.Embedding - 词嵌入（把字符索引转成向量）
3. nn.LayerNorm - 层归一化（稳定训练）
4. nn.Dropout - 随机丢弃（防止过拟合）
5. nn.GELU - 激活函数

下一课我们会详细讲解反向传播的原理。
    """)

if __name__ == "__main__":
    main()
