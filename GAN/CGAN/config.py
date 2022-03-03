# -*- coding: utf-8 -*-
# @Time    : 2021/11/24 11:08
# @Author  : Fisher
# @File    : config.py
# @Software: PyCharm
# @Desc    : 模型相关配置文件

import torch


class Config:

    def __init__(self):
        self.input_size = 110
        self.num_feature = 56 * 56
        self.batch_size = 128
        self.epochs = 100
        self.label_dim = 10
        self.noise_dim = 100
        self.gepochs = 5
        self.model_path = './source'

        self.img_save_path = './result'
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # self.device = torch.device('cpu')


if __name__ == '__main__':
    config = Config()
    print(config.device)
