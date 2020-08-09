# -*- coding: utf-8 -*-

import os
import numpy as np
import torch
import random
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from my_dataset import RMBDataset
from PIL import Image
from matplotlib import pyplot as plt


def set_seed(seed=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


set_seed(1)  # 设置随机种子

# 参数设置
MAX_EPOCH = 10
BATCH_SIZE = 1
LR = 0.01
log_interval = 10
val_interval = 1
rmb_label = {"1": 0, "100": 1}


def transform_invert(img_, transform_train):
    """
    将data 进行反transfrom操作
    :param img_: tensor
    :param transform_train: torchvision.transforms
    :return: PIL image
    """
    if 'Normalize' in str(transform_train):
        norm_transform = list(filter(lambda x: isinstance(x, transforms.Normalize), transform_train.transforms))
        mean = torch.tensor(norm_transform[0].mean, dtype=img_.dtype, device=img_.device)
        std = torch.tensor(norm_transform[0].std, dtype=img_.dtype, device=img_.device)
        img_.mul_(std[:, None, None]).add_(mean[:, None, None])

    # H 表示图像在竖直方向有多少像素，W 表示水平方向像素数，C 表示通道数
    img_ = img_.transpose(0, 2).transpose(0, 1)  # C*H*W --> H*W*C
    img_ = np.array(img_) * 255

    if img_.shape[2] == 3:#彩色图像
        img_ = Image.fromarray(img_.astype('uint8')).convert('RGB')
    elif img_.shape[2] == 1:#灰度图像
        img_ = Image.fromarray(img_.astype('uint8').squeeze())
    else:
        raise Exception("Invalid img shape, expected 1 or 3 in axis 2.数据增强方法, but got {}!".format(img_.shape[2]) )

    return img_


# ============================ step 1/5 数据 ============================
split_dir = os.path.join("rmb_split")
train_dir = os.path.join(split_dir, "train")
valid_dir = os.path.join(split_dir, "valid")

norm_mean = [0.485, 0.456, 0.406]
norm_std = [0.229, 0.224, 0.225]


train_transform = transforms.Compose([
    transforms.Resize((224, 224)),

    # 1 CenterCrop
    # transforms.CenterCrop(512),     # 512

    # 2.数据增强方法 RandomCrop
    # transforms.RandomCrop(224, padding=16),
    # transforms.RandomCrop(224, padding=(16, 64)),
    # transforms.RandomCrop(224, padding=16, fill=(255, 0, 0)),#设置红色padding
    # transforms.RandomCrop(512, pad_if_needed=True),   # pad_if_needed=True   size大于图像尺寸时必须打开
    # transforms.RandomCrop(224, padding=64, padding_mode='edge'),
    # transforms.RandomCrop(224, padding=64, padding_mode='reflect'),
    # transforms.RandomCrop(1024, padding=1024, padding_mode='symmetric'),

    # 3 RandomResizedCrop
    # transforms.RandomResizedCrop(size=224, scale=(0.5, 0.5)),

    # 4 FiveCrop
    # transforms.FiveCrop(112),
    # transforms.Lambda(lambda crops: torch.stack([(transforms.ToTensor()(crop)) for crop in crops])),

    # 5 TenCrop
    # transforms.TenCrop(112, vertical_flip=False),
    # transforms.Lambda(lambda crops: torch.stack([(transforms.ToTensor()(crop)) for crop in crops])),

    # 1 Horizontal Flip
    # transforms.RandomHorizontalFlip(p=1),

    # 2.数据增强方法 Vertical Flip
    # transforms.RandomVerticalFlip(p=0.5),

    # 3 RandomRotation
    # transforms.RandomRotation(90),
    # transforms.RandomRotation((90), expand=True),
    # transforms.RandomRotation(30, center=(0, 0)),
    transforms.RandomRotation(30, center=(0, 0), expand=True),   # expand only for center rotation

    transforms.ToTensor(),
    transforms.Normalize(norm_mean, norm_std),
])

valid_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(norm_mean, norm_std)
])

# 构建MyDataset实例
train_data = RMBDataset(data_dir=train_dir, transform=train_transform)
valid_data = RMBDataset(data_dir=valid_dir, transform=valid_transform)

# 构建DataLoder
train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=False)
valid_loader = DataLoader(dataset=valid_data, batch_size=BATCH_SIZE)


# ============================ step 5/5 训练 ============================
for epoch in range(MAX_EPOCH):
    for i, data in enumerate(train_loader):

        '''
        inputs的形状是一个[1,3,196,196]的形式。第一个维度是size，因为在代码开始设置了BATCH_SIZE=1，
        所以inputs中的第一个维度为1，代表BITCH_SIZE；第二个维度是channel，也就是通道，
        由于是rgb图像，通道的长度为3；第三维和第四位分别是图像的高和宽。
        '''
        inputs, labels = data   # B C H W

        '''
        由于可视化图片是一个三通道的三维张量，所以需要对inputs进行操作，进行索引，索引出第一块区域，
        也就是接下来的一句代码“img_tensor = inputs[0, …]”，这段代码的意思是取四维张量中的第一个三维张量，
        这样就把四维张量变为三维张量了
        '''
        img_tensor = inputs[0, ...]     # C H W
        img = transform_invert(img_tensor, train_transform) #对tansforms进行逆操作--可观察数据输入的样子
        plt.imshow(img)
        plt.show()
        plt.pause(0.5)
        plt.close()

        # bs, ncrops, c, h, w = inputs.shape
        # for n in range(ncrops):
        #     img_tensor = inputs[0, n, ...]  # C H W
        #     img = transform_invert(img_tensor, train_transform)
        #     plt.imshow(img)
        #     plt.show()
        #     plt.pause(1)






