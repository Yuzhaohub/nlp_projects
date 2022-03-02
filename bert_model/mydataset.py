# -*- coding: utf-8 -*-
"""
  @Date    : 2021/10/27 16:50
  @Author  : fisher
  @File    : mydataset
  @Software: PyCharm
  @desc: bert_model数据处理与封装
"""

import re
import math
import torch
import numpy as np
from random import *
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data

text = (
    'Hello, how are you? I am Romeo.\n'  # R
    'Hello, Romeo My name is Juliet. Nice to meet you.\n'  # J
    'Nice meet you too. How are you today?\n'  # R
    'Great. My baseball team won the competition.\n'  # J
    'Oh Congratulations, Juliet\n'  # R
    'Thank you Romeo\n'  # J
    'Where are you going today?\n'  # R
    'I am going shopping. What about you?\n'  # J
    'I am going to visit my grandmother. she is not very well'  # R
)

sentences = re.sub("[.,!?\\-]", '', text.lower()).split('\n')  # filter '.', ',', '?', '!'
word_list = list(set(" ".join(sentences).split()))
word2idx = {'[PAD]': 0, '[CLS]': 1, '[SEP]': 2, '[MASK]': 3}  # 不认识的单词用什么表示：[MASK]?
for i, w in enumerate(word_list):
    word2idx[w] = i + 4
idx2word = {w: i for i, w in word2idx.items()}
vocab_size = len(word2idx)

token_list = list()
for sentence in sentences:
    arr = [word2idx[s] for s in sentence.split()]
    token_list.append(arr)


def make_data(token_list, word2idx, vocab_size, batch_size, max_pred, maxlen):
    """ 数据预处理:
    根据一定的概率随make或者替换（mask）一句话中15%的token，还需要拼接任意两句话
    流程：
        1、两个句子拼接A、B：[CLS] [A] [SEP] [B] [SEP]
        2、随机抽取15%的字符进行MASK操作,判别min(max_len, len * 15%)，要排查[CLS]、[SEP]
        3、进行padding操作
    """
    batch = []
    positive = negative = 0

    while positive != batch_size / 2 or negative != batch_size / 2:
        token_a_index, token_b_index = randrange(len(sentences)), randrange(len(sentences))
        token_a, token_b = token_list[token_a_index], token_list[token_b_index]

        # Bert训练由A、B两句话组成：[CLS] A [SEP] B [SEP]
        input_ids = [word2idx['[CLS]']] + token_a + [word2idx['[SEP]']] + token_b + [word2idx['[SEP]']]
        # segment_ids编码：来自第一个句子编码为0，来自第二个句子编码为1
        segment_ids = [0] * (1 + len(token_a) + 1) + [1] * (len(token_b) + 1)

        # MASK LM： 随机掩码机制: min(max_pred, len * 0.15)
        n_pred = min(max_pred, max(1, int(len(input_ids) * 0.15)))  # MASK一段话中15%的部分
        cand_masked_pos = [i for i, token in enumerate(input_ids)
                           if token != word2idx['[CLS]'] and token != word2idx['[SEP]']]  # 不对[CLS]和[SEP]进行mask操作
        shuffle(cand_masked_pos)

        masked_tokens, masked_pos = [], []  # masked_tokens: 屏蔽的字索引，mask_pos:屏蔽的位置（在句子）
        for pos in cand_masked_pos[:n_pred]:  # pos：索引位置
            masked_pos.append(pos)  # MASK处理字符串的索引
            masked_tokens.append(input_ids[pos])
            if random() < 0.8:  # 80%的概率替换成[MASK]
                input_ids[pos] = word2idx['[MASK]']
            elif random() > 0.9:  # 10%概率随机替换
                index = randint(0, vocab_size - 1)
                while index < 4:  # 不能包含'[CLS]'、'[SEP]'、'[PAD]'
                    index = randint(0, vocab_size - 1)
                input_ids[pos] = index

        # Zero Padddings: 先进行Mask处理，然后pad；A、B合并后封装
        n_pad = maxlen - len(input_ids)
        input_ids.extend([0] * n_pad)
        segment_ids.extend([0] * n_pad)

        # Zero Padding（100% - 15%）tokens
        if max_pred > n_pred:
            n_pad = max_pred - n_pred
            masked_tokens.extend([0] * n_pad)
            masked_pos.extend([0] * n_pad)  # 0表示[CLS]

        if token_a_index + 1 == token_b_index and positive < batch_size / 2:
            batch.append([input_ids, segment_ids, masked_tokens, masked_pos, True])
            positive += 1
        elif token_a_index + 1 != token_b_index and negative < batch_size / 2:
            batch.append([input_ids, segment_ids, masked_tokens, masked_pos, False])
            negative += 1

    return batch


# 数据封装
class MyDataSet(Data.Dataset):
    """ 数据封装 """

    def __init__(self, input_ids, segment_ids, masked_tokens, masked_pos, isNext):
        super(MyDataSet, self).__init__()
        self.input_ids = input_ids
        self.segment_ids = segment_ids
        self.mask_tokens = masked_tokens
        self.mask_pos = masked_pos
        self.isNext = isNext

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.segment_ids[idx], self.mask_tokens[idx], self.mask_pos[idx], self.isNext[idx]


def collate_fn(batch):
    """  """
    input_ids = np.array([item[0] for item in batch], np.int32)
    segment_ids = np.array([item[1] for item in batch], np.int32)
    mask_tokens = np.array([item[2] for item in batch], np.int32)
    mask_pos = np.array([item[3] for item in batch], np.int32)
    isNext = np.array([item[4] for item in batch], np.int32)

    return [
        torch.LongTensor(input_ids),
        torch.LongTensor(segment_ids),
        torch.LongTensor(mask_tokens),
        torch.LongTensor(mask_pos),
        torch.LongTensor(isNext)
    ]


if __name__ == '__main__':
    max_pred = 5
    maxlen = 30
    batch_size = 6

    batch = make_data(token_list = token_list, word2idx = word2idx,
                      vocab_size = vocab_size,
                      batch_size = batch_size, max_pred = max_pred,
                      maxlen = maxlen)
    loader = Data.DataLoader(MyDataSet(*zip(*batch)),
                             collate_fn = collate_fn,
                             batch_size = batch_size)
    print([bt for bt in loader])
