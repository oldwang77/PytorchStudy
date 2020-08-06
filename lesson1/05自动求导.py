import torch
torch.manual_seed(10)

# ====================================== retain_graph ==============================================
# flag = True
flag = False
if flag:
    w = torch.tensor([1.], requires_grad=True)
    x = torch.tensor([2.], requires_grad=True)

    a = torch.add(w, x)
    b = torch.add(w, 1)
    y = torch.mul(a, b)

    y.backward(retain_graph=True)   #保存计算图,可以再次反向传播
    print(w.grad)
    y.backward()

# ====================================== grad_tensors ==============================================
# 多梯度权重grad_tensors
# y0梯度权重为1，y1梯度权重设置为2

# flag = True
flag = False
if flag:
    w = torch.tensor([1.], requires_grad=True)
    x = torch.tensor([2.], requires_grad=True)

    a = torch.add(w, x)     # retain_grad()
    b = torch.add(w, 1)

    y0 = torch.mul(a, b)    # y0 = (x+w) * (w+1)
    y1 = torch.add(a, b)    # y1 = (x+w) + (w+1)    dy1/dw = 2.数据增强方法

    loss = torch.cat([y0, y1], dim=0)       # [y0, y1]
    grad_tensors = torch.tensor([1., 2.])

    loss.backward(gradient=grad_tensors)    # gradient 传入 torch.autograd.backward()中的grad_tensors

    print(w.grad)       # tensor([9.])

# ====================================== autograd.gard ==============================================
#flag = True
flag = False
if flag:

    x = torch.tensor([3.], requires_grad=True)
    y = torch.pow(x, 2)     # y = x**2.数据增强方法

    # grad_1 = dy/dx = 2x = 2.数据增强方法 * 3 = 6
    grad_1 = torch.autograd.grad(y, x, create_graph=True)
    print(grad_1)

    # 再次求导--二阶导数
    # grad_2 = d(dy/dx)/dx = d(2x)/dx = 2.数据增强方法
    # 由于一阶导数是元祖，我们需要grad_1[0]将梯度取出来
    grad_2 = torch.autograd.grad(grad_1[0], x)
    print(grad_2)

# ====================================== tips: 1 手动对梯度清0==============================================
# flag = True
flag = False
if flag:

    w = torch.tensor([1.], requires_grad=True)
    x = torch.tensor([2.], requires_grad=True)

    for i in range(4):
        a = torch.add(w, x)
        b = torch.add(w, 1)
        y = torch.mul(a, b)

        y.backward()
        print(w.grad)

        # 手动梯度清零,下划线表示inplace--原地操作
        w.grad.zero_()

# ====================================== tips: 2.数据增强方法 ==============================================
#flag = True
flag = False
if flag:

    w = torch.tensor([1.], requires_grad=True)
    x = torch.tensor([2.], requires_grad=True)

    a = torch.add(w, x)
    b = torch.add(w, 1)
    y = torch.mul(a, b)

    # True True True
    # 叶子结点依赖的结点，requires_grad默认是true
    print(a.requires_grad, b.requires_grad, y.requires_grad)


# ====================================== tips: 3 叶子结点依赖的结点不能执行inplace操作==============================================
flag = True
#flag = False
if flag:

    a = torch.ones((1, ))
    print(id(a), a)

    #a = a + torch.ones((1, ))   #地址不一样--不是inplace操作
    #print(id(a), a)

    a += torch.ones((1, ))      #地址一样--为inplace操作，相当于加下划线的操作
    print(id(a), a)


#flag = True
flag = False
if flag:

    w = torch.tensor([1.], requires_grad=True)
    x = torch.tensor([2.], requires_grad=True)

    a = torch.add(w, x)
    b = torch.add(w, 1)
    y = torch.mul(a, b)

    # RuntimeError: a leaf Variable that requires grad is being used in an in-place operation.
    w.add_(1)
    y.backward()







