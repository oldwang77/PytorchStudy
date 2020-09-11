# -*- coding: utf-8 -*-
import os
import random
import shutil


def makedir(new_dir):
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)


if __name__ == '__main__':

    # 为了让后面的随机数真正生效
    random.seed(1)

    dataset_dir = os.path.join("..", "..", "data", "RMB_data")  # ../ 表示当前文件所在的目录的上一级目录
    split_dir = os.path.join("rmb_split")
    train_dir = os.path.join(split_dir, "train")
    valid_dir = os.path.join(split_dir, "valid")
    test_dir = os.path.join(split_dir, "test")

    train_pct = 0.8  # 训练集样本比例
    valid_pct = 0.1  # 验证集样本比例
    test_pct = 0.1  # 测试集样本比例

    """
    os.walk方法，主要用来遍历一个目录内各个子目录和子文件。
    os.walk(dataset_dir)：得到一个三元tupple(dirpath, dirnames, filenames)
        dirpath：为起始路径，是一个string，代表目录的路径
        dirnames：为起始路径下的文件夹，是一个list，包含了dirpath下所有子目录的名字
        filenames：是起始路径下的文件，是一个list，包含了非目录文件的名字
    """
    # root:../../data/RMB_data
    # dir:该root目录下所有子目录的名字,即1和100
    # files:该root目录下所有非目录文件的名字，这里没有
    for root, dirs, files in os.walk(dataset_dir):
        # sub_dir:1和100
        for sub_dir in dirs:

            # os.listdir() 方法用于返回指定的文件夹包含的文件或文件夹的名字的列表
            imgs = os.listdir(os.path.join(root, sub_dir))
            # filter是过滤器，过滤出满足尾部条件是.jpg的图片，并且存放在list中
            imgs = list(filter(lambda x: x.endswith('.jpg'), imgs))
            random.shuffle(imgs)  # 打乱列表顺序
            img_count = len(imgs)

            train_point = int(img_count * train_pct)
            valid_point = int(img_count * (train_pct + valid_pct))

            for i in range(img_count):
                if i < train_point:
                    out_dir = os.path.join(train_dir, sub_dir)
                elif i < valid_point:
                    out_dir = os.path.join(valid_dir, sub_dir)
                else:
                    out_dir = os.path.join(test_dir, sub_dir)

                makedir(out_dir)

                target_path = os.path.join(out_dir, imgs[i])            # 图片拷贝的目标路径
                src_path = os.path.join(dataset_dir, sub_dir, imgs[i])  # 图片拷贝的源路径

                shutil.copy(src_path, target_path)                      # 图片从一个路径拷贝到另一个路径

            print('Class:{}, train:{}, valid:{}, test:{}'.format(sub_dir, train_point, valid_point - train_point,
                                                                 img_count - valid_point))
