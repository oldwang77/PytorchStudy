import torch
import numpy as np

# 通过torch.tensor创建张量
flag = True

if flag:
    arr = np.ones((3,3))
    print("ndarray的数据类型",arr.dtype)

    t = torch.tensor(arr)
    print("torch.tensor:",t)

# 通过torch.from_numpy创建张量
# 此时arr和numpy是共享内存的
flag = False
if not flag:
    arr = np.array([[1,2,3],[4,5,6]])
    t = torch.from_numpy(arr)
    print("torch.from_numpy:",arr)


# 通过torch.arrange(start,end,step)
t = torch.arange(1,10,2)
print("torch.arange:",t)

# 通过torch.linspace创建张量
t = torch.linspace(2,10,5)
print("torch.linspace:",t)

# torch.normal(均值,标准差),根据正态分布创建张量
t = torch.arange(0,1)
print("normal:",t)

# torch.rand()
# torch.rand_like() 在【0,1)上，均匀分布