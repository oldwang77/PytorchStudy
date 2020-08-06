# 计算图是用来描述运算的有向无环图
# 两个主要元素，结点（数据）和边（运算）
import torch

w = torch.tensor([1.], requires_grad=True)
x = torch.tensor([2.], requires_grad=True)

# y = (x+w)*(w+1)   x = 2, y = 1
a = torch.add(w, x)
a.retain_grad()

b = torch.add(w, 1)
y = torch.mul(a, b)

y.backward()
print(w.grad)  # tensor([5.])

# 查看叶子结点
print("is_leaf:\n", w.is_leaf, x.is_leaf, a.is_leaf, b.is_leaf, y.is_leaf)

# 查看梯度
# 反向传播结束后，非叶子结点的梯度就释放了，如果想使用非叶子结点的梯度，需要在前面加上a.retain_grad()
print("gradient:\n", w.grad, x.grad, a.grad, b.grad, y.grad)

# 查看 grad_fn 记录创建张量时候的方法
print("grad_fn:\n", w.grad_fn, x.grad_fn, a.grad_fn, b.grad_fn, y.grad_fn)
