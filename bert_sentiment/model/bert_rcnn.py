# -*- coding: utf-8 -*-
# @Time    : 2021/11/3 15:32
# @Author  : Fisher
# @File    : bert_rcnn.py
# @Software: PyCharm
# @Desc    : bert+rcnn模型（bert + rnn + cnn）


import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertTokenizer, BertConfig
from pathlib import Path
from bert_sentiment.albert_model.bert_base import bert_config


class Config:
    """ 配置参数 """

    def __init__(self):
        dataset = Path(__file__).parent.parent.__str__()




if __name__ == '__main__':
    a = torch.randn((64, 30, 200))
    max_pool = nn.MaxPool1d(30)
    out = max_pool(a.permute(0, 2, 1))
    print(out.shape)
