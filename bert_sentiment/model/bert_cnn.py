# -*- coding: utf-8 -*-
# @Time    : 2021/11/2 16:23
# @Author  : Fisher
# @File    : bert_cnn.py
# @Software: PyCharm
# @Desc    : bert+CNN


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


class Model(nn.Module):
    """ bert_cnn模型 """

    def __init__(self, config):
        super(Model, self).__init__()
        self.bert = BertModel.from_pretrained(config.bert_config['bert_dir'])
        for param in self.bert.parameters():
            param.requires_grad = True

        # nn.ModuleList([])：将模型的参数加载到模型上
        self.convs = nn.ModuleList(
            [nn.Conv2d(in_channels=1,
                       out_channels=config.num_filter,
                       kernel_size=(k, config.hidden_size))
             for k in config.filter_size]
        )
        self.dropout = nn.Dropout(config.dropout)
        self.fc_cnn = nn.Linear(config.num_filter * len(config.filter_size), config.num_classes)

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, x):
        """
        x: (x, seq_len, mask), y
        """
        content = x[0]  # 输入的句子
        mask = x[2]  # 对padding部分进行mask，和句子一个size，padding部分用0表示，如：[1, 1,1,1,0,0]

        # 修改BERT模型的配置信息：output_hidden_states\output_attention输出完整的编码向量信息
        # model_config = BertConfig.from_pretrained(config.bert_config['bert_config_path'])
        # model_config.output_hidden_states = True
        # model_config.output_attentions = True
        # bert = BertModel.from_pretrained(config = model_config)
        last_layer_state, pooled = self.bert(input_ids=content, attention_mask=mask)
        out = last_layer_state.unsqueeze(
            1)  # [batch_size, seq_len, hidden_dim] --> [batch_size, 1, seq_len, hidden_dim]

        # [batch_size, out_channels, convs_size, 1] --> max_pool1d处理后：[batch_size, out_channels]
        out = torch.cat([self.conv_and_pool(out, conv) for conv in self.convs], 1)  # [batch_size, out_channels * 3]
        out = self.dropout(out)
        out = self.fc_cnn(out)  # [batch_size, num_classes]
        return out


if __name__ == '__main__':
    from bert_sentiment.utils import build_dataset, build_iterator

    config = Config()
    train, dev, test = build_dataset(config=config)
    data_iter = build_iterator(train, config)
    next(data_iter)

    model = Model(config=config)
    for i, batch in enumerate(data_iter):
        if i == 0:
            x, y = batch
            out = model(x)
            print(out.shape)  # [batch_size, num_classes]
