# -*- coding: utf-8 -*-
# @Time    : 2021/11/17 18:03
# @Author  : Fisher
# @File    : data_loader.py
# @Software: PyCharm
# @Desc    : BERT模型：数据处理与封装

"""
数据处理过程：
    1、数据拼接：[CLS] + context1 + [SEP] + context2 + [SEP]
        传入数据说明：
            input_ids：文本序列
            segment_ids: [0] * len(context1) + [1] * len(context2)
            input_mask: 仅仅mask文本中的padding部分
    2、数据封装：
        DataSet类型
        DataLoader封装器
"""
import numpy as np
import torch
import random
import torch.nn as nn
import torch.utils.data as Data
import re


class Config:
    def __init__(self, word2idx):
        self.word2idx = word2idx
        self.vocab_size = 300
        self.batch_size = 6
        self.max_pred = 5
        self.maxlen = 50


def make_data(token_list, config):
    """ 数据处理

    注意：
        1、MASK处理时，要提出[CLS]和[SEP]
        2、segment_ids: [CLS] + context1 + [SEP] 全部用0表示， context2 + [SEP] 全部用1表示
        3、input_mask: 仅针对padding部分，即[CLS]、[SEP]都不属于mask范畴

    Return:
        input_ids: 文本序列
        segment_ids: 句子标识
        input_mask: padding掩码
    """
    data = []
    positive = negative = 0

    while positive != config.batch_size / 2 or negative != config.batch_size / 2:
        a_index, b_index = random.randrange(len(token_list)), random.randrange(len(token_list))
        token_a, token_b = token_list[a_index], token_list[b_index]

        input_ids = [config.word2idx['[CLS]']] + token_a + [config.word2idx['[SEP]']] + token_b + [
            config.word2idx['[SEP]']]
        segment_ids = [0] + [0] * (1 + len(token_a) + 1) + [1] * (len(token_b) + 1)

        n_pred = min(config.max_pred, max(1, int(len(input_ids) * 0.15)))
        cand_mask_pos = [i for i, word in enumerate(input_ids)
                         if word != config.word2idx['[CLS]'] and word != config.word2idx['[SEP]']]
        random.shuffle(cand_mask_pos)

        masked_tokens, masked_pos = [], []  # 保存MASK屏蔽的词和对应的位置
        for pos in cand_mask_pos[:n_pred]:
            masked_pos.append(pos)
            masked_tokens.append(input_ids[pos])
            rand = random.random()
            if rand < 0.8:
                input_ids[pos] = config.word2idx['[MASK]']
            elif rand < 0.9:
                index = random.randint(0, config.vocab_size - 1)
                while index < 4:  # 排除[CLS]、[SEP]、[PAD]
                    index = random.randint(0, config.vocab_size - 1)
                input_ids[pos] = index

        # 进行padding mask
        n_pad = config.maxlen - len(input_ids)
        input_ids.extend([0] * n_pad)
        segment_ids.extend([0] * n_pad)

        # Zero Padding（100% - 15%）tokens
        if config.max_pred > n_pred:
            n_pad = config.max_pred - n_pred
            masked_tokens.extend([0] * n_pad)
            masked_pos.extend([0] * n_pad)  # 0表示[CLS]

        if a_index + 1 == b_index and positive < config.batch_size / 2:
            data.append([input_ids, segment_ids, masked_tokens, masked_pos, True])
            positive += 1
        elif a_index + 1 != b_index and negative < config.batch_size / 2:
            data.append([input_ids, segment_ids, masked_tokens, masked_pos, False])
            negative += 1
    return data


class MyDataSet(Data.Dataset):
    """ 数据封装 """

    def __init__(self, input_ids, segment_ids, masked_tokens, masked_pos, isNext):
        super(MyDataSet, self).__init__()
        self.input_ids = input_ids
        self.segment_ids = segment_ids
        self.masked_tokens = masked_tokens
        self.masked_pos = masked_pos
        self.isNext = isNext

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.segment_ids[idx], self.masked_tokens[idx], self.masked_pos[idx], self.isNext[
            idx]


def collate_fn(batch):
    """ 针对每一个批次内的数据进行操作
    输入的是：DataSet和DataLoader封装后的对象，以DataSet封装成一个example
    最后以该函数转换后的格式返回一个batch，进行调用
    """

    input_ids = np.array([item[0] for item in batch], np.int32)
    segment_ids = np.array([item[1] for item in batch], np.int32)
    masked_tokens = np.array([item[2] for item in batch], np.int32)
    masked_pos = np.array([item[3] for item in batch], np.int32)
    isNext = np.array([item[4] for item in batch], np.int32)

    return [
        torch.LongTensor(input_ids),
        torch.LongTensor(segment_ids),
        torch.LongTensor(masked_tokens),
        torch.LongTensor(masked_pos),
        torch.LongTensor(isNext)
    ]


if __name__ == '__main__':

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

    config = Config(word2idx)

    batch = make_data(token_list, config)
    loader = Data.DataLoader(MyDataSet(*zip(*batch)),
                             collate_fn=collate_fn,
                             batch_size=config.batch_size)
    print([bt for bt in loader])
