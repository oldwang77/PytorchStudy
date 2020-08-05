import torch


# tensor([[1., 1., 1.],
#         [1., 1., 1.]])
t = torch.ones((2, 3))

# torch.cat，将torch进行一个拼接

t_0 = torch.cat([t, t], dim=0)
t_1 = torch.cat([t, t], dim=1)
print(t_0.shape, t_1.shape)         # torch.Size([4, 3]) torch.Size([2, 6])

# torch.split，将torch进行一个拆分
t = torch.ones(2, 5)
list_of_tensors = torch.split(t, [2, 1, 2], dim=1)
for idx, t in enumerate(list_of_tensors):
    print("第{}个张量:{}，shape is {}".format(idx + 1, t, t.shape))

# torch.transpose,交换张量的两个维度dim0和dim1
t = torch.rand((2,3,4))
t_transpose = torch.transpose(t,dim0=1,dim1=2)
print(t.shape,t_transpose.shape)

# torch.t(),二维张量转置，相当于tanspose(t,dim0=0,dim1=1)

# torch.squeeze 压缩长度为1的维度
# torch.unsqueeze 根据dim进行维度扩展

