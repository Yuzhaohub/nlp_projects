# -*- coding: utf-8 -*-
# @Time    : 2021/11/3 14:48
# @Author  : Fisher
# @File    : bert_dpcnn_brlock.py
# @Software: PyCharm
# @Desc    : DPCNN模型分块


"""
问题：
    1、关于模型参数的问题：为什么有时候需要使用nn.ModuleList()将参数部分绑定到模型上？
        1、设置到有向的图链式计算，就在反向传播时自动将有向图上的参数进行更新
        2、这样可以将部分模块独立出去：就像Bert模型中12层Encoder结构一样
    2、关于DPCNN模型的理解：
        1、金字塔结构的理解：在于不像传统的进行max_pooled操作来提取特征，而是将不断进行(1/2池化)直到达到max_pooled的效果
        2、可以解决文本长依赖的问题，不断扩充文本的感受野，融合更多更长的文本信息
        3、DPCNN的结构：两层卷积层用于提取low_level的文本信息，（1/2）池化层与两层卷积进行特征压缩和融合。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResnetBlock(nn.Module):
    def __init__(self, channel_size):
        super(ResnetBlock, self).__init__()
        self.channel_size = channel_size
        self.maxpool = nn.Sequential(
            nn.ConstantPad1d(padding=(0, 1), value=0),
            nn.MaxPool1d(kernel_size=3, stride=2)
        )
        self.conv = nn.Sequential(
            nn.BatchNorm1d(num_features=self.channel_size),
            nn.ReLU(),
            nn.Conv1d(self.channel_size, self.channel_size,
                      kernel_size=3, padding=1),
            nn.BatchNorm1d(num_features=self.channel_size),
            nn.ReLU(),
            nn.Conv1d(self.channel_size, self.channel_size,
                      kernel_size=3, padding=1),
        )

    def forward(self, x):
        while x.size(-1) > 2:
            x_shortcut = self.maxpool(x)
            x = self.conv(x_shortcut)
            x = x + x_shortcut
        return x.squeeze()


class Model(nn.Module):
    """ DPCNN模型分块 """

    def __init__(self, kernel_list, num_filters, num_classes):
        super(Model, self).__init__()
        self.kernel_list = kernel_list
        self.num_filters = num_filters
        self.num_classes = num_classes

        self.p_conv = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(in_channels=30, out_channels=self.num_filters, kernel_size=k),
                ResnetBlock(num_filters)  # 控制：包含多少个block，直到达到max_pooled的效果
            ) for k in self.kernel_list
        ])
        self.fc = nn.Linear(num_filters * len(kernel_list), self.num_classes)

    def forward(self, x):
        out = [p_conv(x) for p_conv in self.p_conv]
        out = torch.cat(out, dim=-1)
        out = self.fc(out)
        return out


if __name__ == '__main__':
    x = torch.randn((64, 30, 200)).float()
    model = Model(kernel_list=[3, 4, 5], num_filters=100, num_classes=10)
    out = model(x)
    print(out)
