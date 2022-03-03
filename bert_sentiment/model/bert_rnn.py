# -*- coding: utf-8 -*-
# @Time    : 2021/11/2 17:31
# @Author  : Fisher
# @File    : bert_rnn.py
# @Software: PyCharm
# @Desc    : bert+rnn模型


import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertTokenizer, BertConfig
from pathlib import Path
from bert_sentiment.albert_model.bert_base import bert_config

"""
Bert Fine-tuning:
    特征提取器：将
"""


class Config:
    """ 配置参数 """

    def __init__(self):
        dataset = Path(__file__).parent.parent.__str__()
        self.model_name = 'bert'
        self.train_path = dataset + '/data/train.txt'
        self.dev_path = dataset + '/data/dev.txt'
        self.test_path = dataset + '/data/test.txt'
        self.save_path = dataset + '/saved_dict/' + self.model_name + '.ckpt'
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.require_improvement = 1000  # 若超过1000batch效果还没提升，则提前结束训练

        self.class_list = [x.strip() for x in open(dataset + '/data/class.txt').readlines()]
        self.num_classes = len(self.class_list)  # 类别数

        self.num_epochs = 3  # 迭代次数
        self.batch_size = 128  # mini_batch大小
        self.pad_size = 32  # max_len每句话处理的长度
        self.learning_rate = 5e-5  # 学习率
        self.bert_config = bert_config  # bert模型加载与训练，model.from_pretrained()
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_config['bert_vocab_path'])
        self.hidden_size = 768  # 关于BERT模型各层输入输出维度

        self.filter_size = (2, 3, 4)  # 卷积核尺寸
        self.num_filter = 256  # 卷积核数量（channels通道数量）
        self.dropout = 0.1

        self.rnn_hidden = 768
        self.num_layers = 2


class Model(nn.Module):
    """ bert+rnn模型 """

    def __init__(self, config):
        super(Model, self).__init__()
        self.bert = BertModel.from_pretrained(config.bert_config['bert_dir'])
        for param in self.bert.parameters():  # 更新bert模型参数：param.requires_grad = True
            param.requires_grad = True
        self.lstm = nn.LSTM(input_size=config.hidden_size, hidden_size=config.rnn_hidden, num_layers=config.num_layers,
                            bidirectional=True, batch_first=True, dropout=config.dropout)
        self.dropout = nn.Dropout(config.dropout)
        self.fc_rnn = nn.Linear(config.rnn_hidden * 2, config.num_classes)

    def forward(self, x):
        """
        x: (input_ids, seq_len, mask)
            input_ids: [batch_size, seq_len]
            seq_len: [batch_size]
            mask: [batch_size, seq_len]
        """
        context, mask = x[0], x[2]
        last_layer_state, pooled = self.bert(input_ids=context, attention_mask=mask)

        # out: [batch_size, seq_len, directional * hidden_size]
        # h\c: [num_layer * directional, seq_len, hidden_size]
        out, (h, c) = self.lstm(last_layer_state)  # last_layer_state: [batch_size, seq_len, hidden_size]
        out = self.dropout(out)

        # 将lstm模型最后一个隐含层提取特征作为特征向量
        out = out[:, -1, :]  # [batch_size, hidden_size] 最后一个序列输出的隐含层向量作为特征
        out = self.fc_rnn(out)  # [batch_size, num_classes]
        return out


if __name__ == '__main__':
    from bert_sentiment.utils import build_dataset, build_iterator

    config = Config()
    model = Model(config=config)

    train, dev, test = build_dataset(config=config)
    data_iter = build_iterator(train, config)
    next(data_iter)
    for i, batch in enumerate(data_iter):
        if i == 0:
            x, y = batch
            out = model(x)
            print(out.shape)  # [batch_size, num_classes]
