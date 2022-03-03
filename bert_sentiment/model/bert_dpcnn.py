# -*- coding: utf-8 -*-
# @Time    : 2021/11/2 19:18
# @Author  : Fisher
# @File    : bert_dpcnn.py
# @Software: PyCharm
# @Desc    : bert+dpcnn (深度金字塔卷积神经网络)


import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertTokenizer
from pathlib import Path
from bert_sentiment.albert_model.bert_base import bert_config

"""
DPCNN：于TextCNN相比，将原先
    1、区域嵌入（region embedding）：将textCNN的包含多尺寸卷积滤波器的卷积层的卷积结果称之为区域嵌入（即对一个文本区域/文本片段进行一组卷积操作后生成的embedding）
    2、经过区域（）：经过区域嵌入后，是两层卷积层（等长卷积），以此来提高词位embedding的表示的丰富性
        意义（等长卷积）：将输入序列的每个词位及其左右（（n-1）/2）个词的上下文信息压缩为该词位的embedding（产生了每个词位的被上下文信息修饰过的更高level更加准确的语义）
    3、下采样（1/2池化）：使用size=3，stride=2（大小位3，步长位2）的池化层进行最大化。每经过1/2池化层，序列的长度就被压缩成原来的一半、
    4、固定Feature Map的数量：
        * 文本处理相比较图像处理更容易发生语义取代（即high-level可以被low-level取代），DPCNN固定feature map数量，也就是固定了embedding space的维度（语义空间）
        * 使得网络有可能让整个邻接词（邻接ngram）的合并操作在原始空间或者与原始空间相似的空间中进行。
        * 整个网络虽然形状上看是深层的，但是从语义空间上来看完全是扁平的
    5、残差链接：
        * 由于初始化深度CNN时，往往各层权重都是初始化为一个很小的值（这就会导致后续几乎每层的输入都是接近0），这些小权重同时阻碍了梯度的传播，容易发生梯度爆炸或弥散问题
        * 既然每个block的输入在初始阶段容易是0而无法激活，那么直接用一条线把region embedding层链接到每个block的输入乃至最终的池化层/输出层
    6、由于1/2池化层的存在，文本序列的长度会随着block数量的增加呈指数级减少，导致序列长度随着网络加深呈现金字塔（Pyramid）形状。       
"""


class Config(object):
    """配置参数"""

    def __init__(self):
        dataset = Path(__file__).parent.parent.__str__()
        self.model_name = 'bert'
        self.train_path = dataset + '/data/train.txt'  # 训练集
        self.dev_path = dataset + '/data/dev.txt'  # 验证集
        self.test_path = dataset + '/data/test.txt'  # 测试集
        self.class_list = [x.strip() for x in open(
            dataset + '/data/class.txt').readlines()]  # 类别名单
        self.save_path = dataset + '/saved_dict/' + self.model_name + '.ckpt'  # 模型训练结果
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 设备
        self.require_improvement = 1000  # 若超过1000batch效果还没提升，则提前结束训练
        self.num_classes = len(self.class_list)  # 类别数
        self.num_epochs = 3  # epoch数
        self.batch_size = 128  # mini-batch大小
        self.pad_size = 32  # 每句话处理成的长度(短填长切)
        self.learning_rate = 5e-5  # 学习率

        # 注意路径的问题：bertmodel导入模型需要绝对路径
        self.bert_config = bert_config
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_config['bert_vocab_path'])
        self.hidden_size = 768
        self.num_filters = 250  # 卷积核数量(channels数)


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.bert = BertModel.from_pretrained(config.bert_config['bert_dir'])  # bert模型加载使用绝对路径：
        for param in self.bert.parameters():
            param.requires_grad = True
        # self.fc = nn.Linear(config.hidden_size, config.num_classes)
        self.conv_region = nn.Conv2d(1, config.num_filters, (3, config.hidden_size), stride=1)
        self.conv = nn.Conv2d(config.num_filters, config.num_filters, (3, 1), stride=1)
        self.max_pool = nn.MaxPool2d(kernel_size=(3, 1), stride=2)

        #  nn.ZeroPad2d(left, right, top, bottom)：边界填充num零向量
        self.padding1 = nn.ZeroPad2d((0, 0, 1, 1))  # top bottom
        self.padding2 = nn.ZeroPad2d((0, 0, 0, 1))  # bottom
        self.relu = nn.ReLU()
        self.fc = nn.Linear(config.num_filters, config.num_classes)

    def forward(self, x):
        """ 未使用多尺寸滤波器的卷积层 """
        context = x[0]  # 输入的句子
        mask = x[2]  # 对padding部分进行mask，和句子一个size，padding部分用0表示，如：[1, 1, 1, 1, 0, 0]
        encoder_out, text_cls = self.bert(context, attention_mask=mask)
        x = encoder_out.unsqueeze(1)  # [batch_size, 1, seq_len, embed]

        # 多尺寸滤波器的卷积层：更改conv_region层，在该部分进行循环
        # nn.ModuleList([nn.Sequential() for k in [3, 4, 5]])

        x = self.conv_region(x)  # [batch_size, 250, seq_len-3+1, 1]
        x = self.padding1(x)  # [batch_size, 250, seq_len, 1]
        x = self.relu(x)  # 卷积层后，接激活层（将线性空间映射到其他空间）
        x = self.conv(x)  # [batch_size, 250, seq_len-3+1, 1]
        x = self.padding1(x)  # [batch_size, 250, seq_len, 1]
        x = self.relu(x)
        x = self.conv(x)  # [batch_size, 250, seq_len-3+1, 1]

        # block: 进行卷积核池化（1/2池化），不再简单的进行对low-level层进行max_pooled，而是不断卷积池化操作，完成max_pooled
        while x.size()[2] > 2:
            x = self._block(x)

        # 第二种方式进行金字塔特征提取：nn.Sequential()，将其放置到初始化部分，构建模型
        # seq_len = 100
        # resent_block_list = []
        # while seq_len > 2:
        #     resent_block_list.append(self._block(''))
        # self.resnet_layer = nn.Sequential(*resent_block_list)

        x = x.squeeze()  # [batch_size, num_filter(250), 1, 1] --> [batch_size, num_filters(250)]
        x = self.fc(x)
        return x

    def _block(self, x):
        """
        每次卷积操作后：维度 - 2,
        每次pad操作后： 维度 + 2
        池化（1/2池化）--> 激活层-->卷积层-->激活层-->卷积层-->残差拼接层
        """
        x = self.padding2(x)  # [batch_size, 250, seq_len - 1, 1]
        px = self.max_pool(x)  # [batch_size, 250, (seq_len - 1)//2, 1]
        x = self.padding1(px)  # [batch_size, 250, (seq_len - 1)//2 + 2, 1]
        x = F.relu(x)
        x = self.conv(x)  # [batch_size, 250, (seq_len - 1) // 2, 1]
        x = self.padding1(x)
        x = F.relu(x)
        x = self.conv(x)
        x = x + px  # short cut: 残差链接：为防止梯度消失，第一次池化后的向量+池化卷积（两层）后的向量
        return x


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
