# -*- coding: utf-8 -*-
# @Time    : 2021/11/2 11:14
# @Author  : Fisher
# @File    : model.py
# @Software: PyCharm


import os
import torch.nn as nn
import torch
from transformers import BertModel, BertTokenizer, BertConfig
from bert_sentiment.albert_model.bert_base import bert_config
from pathlib import Path


class Config(object):
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


class Model(nn.Module):
    """ BERT模型 """

    def __init__(self, config):
        super(Model, self).__init__()
        self.bert = BertModel.from_pretrained(config.bert_config['bert_dir'])
        for param in self.bert.parameters():
            param.requires_grad = True
        self.fc = nn.Linear(config.hidden_size, config.num_classes)

    def forward(self, x):
        """
        Output: tuple对象，包括四层编码向量：last_hidden_state、pooler_output、hidden_states、attentions
            last_hidden_state: 最后一层隐含层的输出向量（总共12层），[batch_size, seq_len, hidden_size]
            pooled_output: 最后一层隐含层的第一个向量([CLS])，经过线性层和Tanh激活层,[batch_size, hidden_size]
            hidden_states: 所有的隐含层（12层）+词嵌入层向量
            attentions: 注意力层（12层）
        Input:
            input_ids: [batch_size, seq_len]，序列编码处理后的文本序列
            attention_mask: [batch_size, seq_len]，对pad进行mask
        """
        context = x[0]  # 输入的句子
        mask = x[2]  # 对padding部分进行mask，和句子一个size，padding部分用0表示，如：[1, 1, ,1, 1, 0, 0]

        # 修改BERT模型的配置信息：output_hidden_states\output_attention输出完整的编码向量信息
        # model_config = BertConfig.from_pretrained(config.bert_config['bert_config_path'])
        # model_config.output_hidden_states = True
        # model_config.output_attentions = True
        # bert = BertModel.from_pretrained(config = model_config)

        last_hidden_state, pooled = self.bert(input_ids=context, attention_mask=mask, output_all_encoded_layers=False)
        out = self.fc(pooled)  # pooled: [CLS]对象的最后一层词向量,[batch_size, seq_len]
        return out


if __name__ == '__main__':
    config = Config()
    print(config.train_path)
    tokenize = config.tokenizer
    s = '[CLS]我是谁'
    s_split = tokenize.tokenize(s)
    print(s_split)
