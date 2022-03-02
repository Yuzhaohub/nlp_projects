"""
  @Date    : 2021/10/18 15:18
  @Author  : fisher
  @File    : mydataset
  @Software: PyCharm
  @desc: 数据格式转换与封装
"""

"""
数据格式：
    T: [batch_size, seq_len]
    S1: [batch_size, seq_len]
    S2: [batch_size, seq_len]
    K1: [batch_size, 1]
    K2: [batch_size, 1]
    O1: [batch_size, seq_len]
    O2: [batch_size, seq_len]
数据疯转： 
    torch.utils.data.Dataset | torch.utils.data.DataLoader
"""

import yaml
import json
import os
import numpy as np
import random
from torch.utils import data
import torch

configsfile = r'data/configs.yaml'


class DataFormat(object):
    """
    数据格式：
    T: 文本序列化, [batch_size, seq_len]
    S1: subject，[batch_size, seq_len]
    S2: [batch_size, seq_len]
    K1: [batch_size, 1]
    K2: [batch_size, 1]
    O1: [batch_size, seq_len]
    O2: [batch_size, seq_len]
    """

    def __init__(self, data, batch_size = 64):
        self.data = data
        self.batch_size = batch_size
        self.steps = len(self.data) // self.batch_size
        if len(self.data) % self.batch_size != 0:
            self.steps += 1
        self.load_configs()

    def __len__(self):
        return self.steps

    def _read(self, filepath):
        """ 读取配置 """
        return json.load(open(filepath, encoding = 'utf-8'))

    def load_configs(self):
        """ 导入配置相关字典表 """
        fopen = open(configsfile)
        configs = yaml.load(fopen, Loader = yaml.FullLoader)

        all_schemas_file = os.path.join(configs['datapath'], configs['all_schemas'])
        all_char_file = os.path.join(configs['datapath'], configs['all_chars'])

        self.schemas2id, _ = self._read(all_schemas_file)
        self.word2id, _ = self._read(all_char_file)

    def seq_padding(self, X, max_len = None):
        """ padding机制（该部分需要修改默认采用最大长度）、mask机制:  """
        if max_len is None:
            L = [len(x) for x in X]
            max_len = max(L)
        return [x + [0] * (max_len - len(x)) for x in X]

    def pre_res(self):
        """ 将所有文本进行序列化 """
        idxs = list(range(len(self.data)))
        T, S1, S2, K1, K2, O1, O2 = [], [], [], [], [], [], []
        np.random.shuffle(idxs)  # np.random
        for i in idxs:
            d = self.data[i]
            text = d['text']
            items = {}  # 抓取一对多的样本： {subject: [[obecjt, object]. []]}
            # 关于subject 一对多 predicate、object的问题
            for spo in d['spo_list']:
                subject_id = text.find(spo[0])
                object_id = text.find(spo[2])
                if subject_id != -1 and object_id != -1:  # s.find(q) 没有匹配到，返回-1
                    key = (subject_id, subject_id + len(spo[0]))
                    if key not in items:
                        items[key] = []
                    items[key].append((object_id, object_id + len(spo[2]), self.schemas2id[spo[1]]))
            if items:
                T.append([self.word2id.get(c, 1) for c in text])  # 1: unk, 0:pad
                s1, s2 = [0] * len(text), [0] * len(text)
                for j in items:  # items.key(): (起始位置, 终止位置)
                    s1[j[0]] = 1
                    s2[j[1] - 1] = 1  # 索引： len(text) - 1
                # 训练数据：随机抽取
                k1, k2 = random.choice(list(items.keys()))
                o1, o2 = [0] * len(text), [0] * len(text)
                for q in items[(k1, k2)]:
                    o1[q[0]] = q[2]    # 多类别编码处理
                    o2[q[1] - 1] = q[2]
                S1.append(s1)
                S2.append(s2)
                K1.append([k1])
                K2.append([k2])
                O1.append(o1)
                O2.append(o2)

        T = np.asarray(self.seq_padding(T))
        S1 = np.asarray(self.seq_padding(S1))
        S2 = np.asarray(self.seq_padding(S2))
        K1 = np.asarray(K1)
        K2 = np.asarray(K2)
        O1 = np.asarray(self.seq_padding(O1))
        O2 = np.asarray(self.seq_padding(O2))

        return [T, S1, S2, K1, K2, O1, O2]


class MydataSet(data.Dataset):
    """
    自定义数据集：
        1、改写方法:__getitem__()
        2、封装: 自定义分批次梳理函数collate_fn
    """

    def __init__(self, _T, _S1, _S2, _K1, _K2, _O1, _O2):
        self._T = _T
        self._S1 = _S1
        self._S2 = _S2
        self._K1 = _K1
        self._K2 = _K2
        self._O1 = _O1
        self._O2 = _O2

        self.len = len(self._T)

    def __getitem__(self, index):
        """ 改写__getitem__ """
        return self._T[index], self._S1[index], self._S2[index], self._K1[index], self._K2[index], self._O1[index], \
               self._O2[index]

    def __len__(self):
        return self.len

    def collate_fn(self, batch):
        """ 自定义分批次处理函数 """
        # TODO np.int32与np.int32的区别：int32会出现一部分数据为负数
        T = np.array([item[0] for item in batch], np.int32)
        S1 = np.array([item[1] for item in batch], np.int32)
        S2 = np.array([item[2] for item in batch], np.int32)
        K1 = np.array([item[3] for item in batch], np.int32)
        K2 = np.array([item[4] for item in batch], np.int32)
        O1 = np.array([item[5] for item in batch], np.int32)
        O2 = np.array([item[6] for item in batch], np.int32)

        return {
            'T': torch.LongTensor(T),
            'S1': torch.FloatTensor(S1),
            'S2': torch.FloatTensor(S2),
            'K1': torch.LongTensor(K1),
            'K2': torch.LongTensor(K2),
            'O1': torch.LongTensor(O1),
            'O2': torch.LongTensor(O2),
        }


class MyDataLoader:
    """ 封装成DataLoader对象 """

    def __init__(self, batch_size):
        self.batch_size = batch_size

    def loader(self, mydataset, collate_fn):
        """ 数据分批次处理 """
        loader = data.DataLoader(
            dataset = mydataset,  # torch TensorDataset format
            batch_size = self.batch_size,  # mini batch size
            shuffle = True,  # random shuffle for training
            collate_fn = collate_fn,  # 如何取样本，自定义样本处理函数
            num_workers = 0
        )
        return loader


if __name__ == '__main__':
    train_data = json.load(open(r'data/trans_data/train_json_me.json', encoding = 'utf-8'))
    mydataformat = DataFormat(train_data)
    res = mydataformat.pre_res()
    mydataset = MydataSet(*res)
    mydataloader = MyDataLoader(batch_size = 32)
    loader = mydataloader.loader(mydataset, mydataset.collate_fn)
    for batch in loader:
        print(batch)
