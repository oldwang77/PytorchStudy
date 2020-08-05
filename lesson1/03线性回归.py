import torch
import matplotlib.pyplot as plt

torch.manual_seed(10)  # 随机数种子

lr = 0.1  # 学习率

# 创建训练数据
x = torch.rand(20, 1) * 10  # x shape(20,1)
y = 2 * x + (5 + torch.rand(20, 1))  # y

# 构造线性回归参数
# requires_grad表示需要计算梯度
w = torch.randn((1), requires_grad=True)
b = torch.ones((1), requires_grad=True)

for iteration in range(1000):
    # 前向传播
    wx = torch.mul(w, x)
    y_pred = torch.add(wx, b)

    # 计算 MSE loss
    loss = (0.5 * (y - y_pred) ** 2).mean()

    # 反向传播
    loss.backward()

    # 更新参数
    b.data.sub_(lr * b.grad)
    w.data.sub_(lr * w.grad)

    # 绘图
    if iteration % 20 == 0:
        plt.scatter(x.data.numpy(), y.data.numpy())
        plt.plot(x.data.numpy(), y_pred.data.numpy())
        plt.pause(0.5)

        if loss.data.numpy() < 1:
            break
