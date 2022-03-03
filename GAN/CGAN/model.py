# -*- coding: utf-8 -*-
# @Time    : 2021/11/24 10:01
# @Author  : Fisher
# @File    : model.py
# @Software: PyCharm
# @Desc    : CGAN模型


import torch.nn as nn
import torch


class Discriminator(nn.Module):
    """ 判别器：结果输出不再是0/1，而是类别的one-hot """

    def __init__(self, config):
        super(Discriminator, self).__init__()
        self.dis = nn.Sequential(
            nn.Conv2d(1, 32, 5, stride = 1, padding = 2),
            nn.LeakyReLU(0.2, True),
            nn.MaxPool2d((2, 2)),

            nn.Conv2d(32, 64, 5, stride = 1, padding = 2),
            nn.LeakyReLU(0.2, True),
            nn.MaxPool2d((2, 2))
        )

        self.fc = nn.Sequential(
            nn.Linear(7 * 7 * 64, 1024),
            nn.LeakyReLU(0.2, True),
            nn.Linear(1024, 10),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        x: [batch_size, 1, 28, 28]
        """
        x = self.dis(x)  # [batch_size, 64, 7, 7]
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x  # [batch_size, 10]


class Generator(nn.Module):
    """ 生成器：输入信息加入标签信息 """

    def __init__(self, config):
        super(Generator, self).__init__()
        self.fc = nn.Linear(config.input_size, config.num_feature)
        self.gen = nn.Sequential(
            nn.BatchNorm2d(1),
            nn.ReLU(True),

            nn.Conv2d(1, 50, 3, stride = 1, padding = 1),
            nn.BatchNorm2d(50),
            nn.ReLU(True),

            nn.Conv2d(50, 25, 3, stride = 1, padding =1),
            nn.BatchNorm2d(25),
            nn.ReLU(True),

            nn.Conv2d(25, 1, 2, stride = 2),
            nn.Tanh()
        )

    def forward(self, x):
        """
        x: [batch_size, 110] 100维的噪声+10维的标签
        """
        x = self.fc(x)
        x = x.view(x.size(0), 1, 56, 56)
        x = self.gen(x)
        return x  # [batch_size, 1, 28, 28]


if __name__ == '__main__':
    class Config:
        input_size = 110
        num_feature = 56 * 56

    config = Config()
    a = torch.randn(32, 110)
    gen = Generator(config)
    gen(a)
