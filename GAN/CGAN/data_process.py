# -*- coding: utf-8 -*-
# @Time    : 2021/11/24 10:39
# @Author  : Fisher
# @File    : data_process.py
# @Software: PyCharm
# @Desc    : 数据处理与封装


from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from config import Config

config = Config()


def load_data(config):
    """ 导入数据 """
    transform_img = transforms.Compose([transforms.ToTensor()])  # 图片类型转换
    trainset = MNIST('./data', train = True, transform = transform_img, download = True)
    trainloader = DataLoader(trainset, batch_size = config.batch_size, shuffle = True)
    return trainloader


# 定义展示图片的函数
def show_images(images):
    """ 展示图片 """
    print('images: ', images.shape)
    images = np.reshape(images, [images.shape[0], -1])
    sqrtn = int(np.ceil(np.sqrt(images.shape[0])))
    sqrtimg = int(np.ceil(np.sqrt(images.shape[1])))

    fig = plt.figure(figsize = (sqrtn, sqrtn))
    gs = gridspec.GridSpec(sqrtn, sqrtn)
    gs.update(wspace = 0.05, hspace = 0.05)

    for i, img in enumerate(images):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(img.reshape([sqrtimg, sqrtimg]))
    plt.show()
    return


def deprocess_img(img):
    out = 0.5 * (img + 1)
    out = out.clamp(0, 1)
    out = out.view(-1, 1, 28, 28)
    return out


if __name__ == '__main__':
    trainloader = load_data(config)
    for item in trainloader:
        print(item)

    imgs = item[0][:10]
    labels = item[1][:10]
    print(labels)
    show_images(imgs)





