# -*- coding: utf-8 -*-
# @Time    : 2021/11/2 14:35
# @Author  : Fisher
# @File    : utils.py
# @Software: PyCharm
# @Desc    : 数据处理以及封装

import torch
from tqdm import tqdm
import time
from datetime import timedelta

"""
Bert模型对单个句子的处理方式：
    1、[CLS] + context + [SPE]
    2、[CLS] + context
"""

PAD, CLS = '[PAD]', '[CLS]'


def build_dataset(config):
    """ 构建数据集 """

    def load_dataset(path, pad_size=None):
        contents = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in tqdm(f):
                line = line.strip()
                if not line:
                    continue
                content, label = line.split('\t')
                token = config.tokenizer.tokenize(content)
                token = [CLS] + token
                seq_len = len(token)
                mask = []
                token_ids = config.tokenizer.convert_tokens_to_ids(token)

                if pad_size is None:
                    pad_size = config.pad_size
                if len(token) < pad_size:
                    mask = [1] * seq_len + [0] * (pad_size - seq_len)
                    token_ids += [0] * (pad_size - seq_len)
                else:
                    token_ids = token_ids[:pad_size]
                    mask = [1] * pad_size
                    seq_len = pad_size
                contents.append((token_ids, int(label), seq_len, mask))
        return contents

    train = load_dataset(config.train_path, config.pad_size)
    dev = load_dataset(config.dev_path, config.pad_size)
    test = load_dataset(config.test_path, config.pad_size)
    return train, dev, test


class DatasetIterater(object):
    """ 数据迭代器 """

    def __init__(self, batches, batch_size, device):
        self.batch_size = batch_size
        self.batches = batches
        self.n_batches = len(batches) // batch_size
        self.residue = False  # 记录batch数量是否为整数
        if len(batches) % self.n_batches != 0:
            self.residue = True

        # 相当于类中的全局变量
        self.index = 0
        self.device = device

    def _to_tensor(self, datas):
        """ 转化tensor对象 """
        x = torch.LongTensor([_[0] for _ in datas]).to(self.device)
        y = torch.LongTensor([_[1] for _ in datas]).to(self.device)

        # pad前的长度（超过pad_size的设为pad_size）
        seq_len = torch.LongTensor([_[2] for _ in datas]).to(self.device)
        mask = torch.LongTensor([_[3] for _ in datas]).to(self.device)
        return (x, seq_len, mask), y

    def __next__(self):
        """ 数据封装处理：构建迭代器改写__next__ 得设置类的全局变量 """
        if self.residue and self.index == self.n_batches:
            batches = self.batches[self.index * self.batch_size: len(self.batches)]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches
        elif self.index >= self.n_batches:
            self.index = 0
            return StopIteration
        else:
            batches = self.batches[self.index * self.batch_size: (self.index + 1) * self.batch_size]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

    def __getitem__(self, item):
        return self._to_tensor(self.batches[item])

    def __iter__(self):
        return self

    def __len__(self):
        if self.residue:
            return self.n_batches + 1
        else:
            return self.n_batches


def build_iterator(dataset, config):
    """ 构建迭代器 """
    iter = DatasetIterater(dataset, config.batch_size, config.device)
    return iter


def get_time_dif(start_time):
    """ 计算时间 """
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))


if __name__ == '__main__':
    from model.bert import Config

    config = Config()
    train, dev, test = build_dataset(config=config)
    data_iter = build_iterator(train, config)
    next(data_iter)
