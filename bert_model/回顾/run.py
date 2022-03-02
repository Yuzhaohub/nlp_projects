# -*- coding: utf-8 -*-
# @Time    : 2021/11/18 15:02
# @Author  : Fisher
# @File    : run.py
# @Software: PyCharm
# @Desc    : Bert模型训练


import torch
import torch.nn as nn
from model import BERT, Config
import torch.optim as optim
from data_loader import make_data, MyDataSet

config = Config()
model = BERT(config)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adadelta(model.parameters(), lr=0.001)


def train(loader, config):
    """ 模型训练 """
    for epoch in range(config.epochs):
        for input_ids, segment_ids, masked_tokens, masked_pos, isNext in loader:
            logits_lm, logits_clsf = model(input_ids, segment_ids, masked_pos)

            # logits_lm: [batch_size, max_pred, vocab_size] -> [batch_size, max_pred * vocab_size]
            loss_lm = criterion(logits_lm.view(-1, config.vocab_size), masked_tokens.view(-1))
            loss_lm = (loss_lm.float()).mean()

            # logits_clsf: [batch_size, 2], isNext: [batch_size]
            loss_clsf = criterion(logits_clsf, isNext)
            loss = loss_clsf + loss_lm
            if (epoch + 1) % 10 == 0:
                print('Epoch:', '%04d' % (epoch + 1), 'loss =', '{:.6f}'.format(loss))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
