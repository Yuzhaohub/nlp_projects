# -*- coding: utf-8 -*-
"""
  @Date    : 2021/10/28 16:53
  @Author  : fisher
  @File    : model
  @Software: PyCharm
  @desc: Bert模型构建
"""

"""
Transformer模型结构说明：
    1、Encoder（编码层）：
        1、Embedding（编码层）：
             Position Encoding（位置嵌入）
             Embedding（词向量嵌入）
             Segment Encoding（句子归属嵌入）
        2、Multi-Head Attention（多头注意力层）
        3、Add & Norm（残差连接、层归一化）
        4、Forward（全连接前馈层）
    2、Decoder（解码层）：
        1、包括两个Multi-Head Attention层。
            1、第一个Multi-Head Attention层采用Masked操作
            2、第二个Multi-Head Attention层的K、V矩阵使用Encoder的编码信息矩阵C进行计算，而Q使用上一个Decoder block的输出计算
        2、最后有一个Softmax层计算下一个翻译单词的概率

"""

import torch
import torch.nn as nn
import math
import numpy as np


# 模型结构主要采用Transformer的Encoder
def get_attn_pad_mask(seq_q, seq_k):
    """
    Padding Mask操作
    :param seq_q: [batch_size, seq_len]
    :param seq_k: [batch_size, seq_len]
    :return: [batch_size, seq_len, seq_len]
    """
    batch_size, seq_len = seq_q.size()
    pad_attn_mask = seq_q.data.eq(0).unsqueeze(1)  # [batch_size, 1, seq_len]
    return pad_attn_mask.expand(batch_size, seq_len, seq_len)


def gelu(x):
    """
    激活函数：
    0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
    x: [batch_size, seq_len, d_model]
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


# TODO 该部分可以通过配置文件导入为全局变量
vocab_size = 200  # 词典表的维度
d_model = 100  # embedding维度
n_segments = 2  # 句子的个数（NSP）
d_k = d_v = 64  # 由embedding --> Q、K、V对应的特征向量维度
n_heads = 12  # 多头注意力机制的个数
d_ff = 768 * 4  # 前向传播的维度
maxlen = 30  # 文本长度
n_layers = 12


class Embedding(nn.Module):
    """ 嵌入层部分 """

    def __init__(self):
        super(Embedding, self).__init__()
        self.tok_embed = nn.Embedding(vocab_size, d_model)  # token embedding
        self.pos_embed = nn.Embedding(maxlen, d_model)  # position embedding
        self.seg_embed = nn.Embedding(n_segments, d_model)  # segment(token type) embedding
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, seg):
        """
        嵌入层：词向量+位置编码+句子从属编码
        :param x: [batch_size, seq_len]
        :param seg: [batch_size, seq_len]
        :return:
        """
        seq_len = x.size(1)
        pos = torch.arange(seq_len, dtype = torch.long)
        pos = pos.unsqueeze(0).expand_as(x)  # [seq_len] --> [batch_size, seq_len]
        embedded = self.tok_embed(x) + self.pos_embed(pos) + self.seg_embed(seg)
        return self.norm(embedded)


class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V, attn_mask):
        """
        attn_mask: 在softmax之前处理进行掩码处理
        :param Q: [batch_size, n_heads, seq_len, d_k]
        :param K: [batch_size, n_heads, seq_len, d_k]
        :param V: [batch_size, n_heads, seq_len, d_v]
        :param attn_mask: [batch_size, n_heads, seq_len, seq_len]
        :return:
        """
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k)  # scores: [batch_size, n_heads, seq_len, seq_len]
        scores.masked_fill_(attn_mask, -1e9)  # 对pad填充负无穷的数据，softmax为零
        attn = nn.Softmax(dim = -1)(scores)  # att: [batch_size, n_heads, seq_len, seq_len]

        # [batch_size, n_heads, seq_len, seq_len] * [batch_size, n_heads, seq_len, d_k] -->
        # [batch_size, n_heads, seq_len, d_k]
        context = torch.matmul(attn, V)
        return context


class MultiHeadAttention(nn.Module):
    """ 多头注意力机制 """

    def __init__(self):
        super(MultiHeadAttention, self).__init__()
        self.W_Q = nn.Linear(d_model, d_k * n_heads)
        self.W_k = nn.Linear(d_model, d_k * n_heads)
        self.W_v = nn.Linear(d_model, d_v * n_heads)

    def forward(self, Q, K, V, attn_mask):
        """
        一般情况下，Q、K、V三者向量相同，即为Embedding处理后
        :param Q: [batch_size, seq_len, d_model]
        :param K: [batch_size, seq_len, d_model]
        :param V: [batch_size, seq_len, d_model]
        :param attn_mask: [batch_size, seq_len, seq_len] 复制n_heads次
        :return:
        """
        residual, batch_size = Q, Q.size(0)
        q_s = self.W_Q(Q).view(batch_size, -1, n_heads, d_k).transpose(1, 2)  # [batch_size, n_heads, seq_len, d_k]
        k_s = self.W_k(K).view(batch_size, -1, n_heads, d_k).transpose(1, 2)  # [batch_size, n_heads, seq_len, d_k]
        v_s = self.W_v(V).view(batch_size, -1, n_heads, d_v).transpose(1, 2)  # [batch_size, n_heads, seq_len, d_v]

        attn_mask = attn_mask.unsqueeze(1).repeat(1, n_heads, 1, 1)  # [batch_size, n_heads, seq_len, seq_len]

        context = ScaledDotProductAttention()(q_s, k_s, v_s, attn_mask)  # [batch_size, n_heads, seq_len, d_v]
        # [batch_size, n_heads, seq_len, d_v] ----> [batch_size, seq_len, n_heads * d_v]
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, n_heads * d_v)  # .contiguous()断开连接

        output = nn.Linear(n_heads * d_v, d_model)(context)  # [batch_size, seq_len, d_model]
        return nn.LayerNorm(d_model)(output + residual)  # [batch_size, seq_len, d_model]


class PoswiseFeedForwardNet(nn.Module):
    """ 前馈神经网络 """

    def __init__(self):
        super(PoswiseFeedForwardNet, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        """
        前馈层：经过词嵌入层、注意力层、层归一化层、残差连接层
        :param x: [batch_size, seq_len, d_model]
        :return:
        """
        return self.fc2(gelu(x))


class EncoderLayer(nn.Module):
    """ encoder block """

    def __init__(self):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention()
        self.pos_ffn = PoswiseFeedForwardNet()

    def forward(self, enc_inputs, enc_self_attn_mask):
        """
        :param enc_inputs: [batch_size, seq_len, d_model]
        :param enc_self_attn_mask:  [batch_size, seq_len, seq_len]
        :return:
        """
        enc_outputs = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs, enc_self_attn_mask)
        enc_outputs = self.pos_ffn(enc_outputs)  # [batch_size, seq_len, d_model]
        return enc_outputs


class BERT(nn.Module):
    def __init__(self):
        super(BERT, self).__init__()
        self.embedding = Embedding()  # 自定义Embedding：word_embed|pos_embed|segment_embed

        # nn.ModuleList: 储存不同module，并自动将每个module的parameter添加到网络之中的容器
        # nn.Sequential内部实现了forward函数，因此可以不用写forward函数。而nn.ModuleList则没有实现内部forward函数。
        self.layers = nn.ModuleList([EncoderLayer() for _ in range(n_layers)])
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Dropout(0.5),
            nn.Tanh()
        )

        self.classifier = nn.Linear(d_model, 2)
        self.linear = nn.Linear(d_model, d_model)
        self.active2 = gelu

        # fc2 is shared with embedding layer
        embed_weight = self.embedding.tok_embed.weight  # nn.Embedding(vocab_size, d_model): [vocab_size, d_model]
        self.fc2 = nn.Linear(d_model, vocab_size, bias = False)
        self.fc2.weight = embed_weight

    def forward(self, input_ids, segment_ids, masked_pos):
        """
        整个流程：嵌入层、注意力、残差连接、层归一化
        :param input_ids:  [batch_size, seq_len]
        :param segment_ids: [batch_size, seq_len]
        :param masked_pos: [batch_size, max_pred]
        :return:
        """
        output = self.embedding(input_ids, segment_ids)  # [batch_size, seq_len, d_model]
        enc_self_attn_mask = get_attn_pad_mask(input_ids, input_ids)  # [batch_size, seq_len, seq_len]
        for layer in self.layers:
            output = layer(output, enc_self_attn_mask)  # [batch_size, seq_len, d_model]

        h_pooled = self.fc(output[:, 0])  # 获取[CLS]的词向量：起始位置 来对是否为下一句进行预测
        logits_clsf = self.classifier(h_pooled)  # [batch_size, 2] 句子分类用：CLS向量

        # 抓取MASK词的特征向量进行训练，该特征向量融合了全局的特征信息。
        masked_pos = masked_pos[:, :, None].expands(-1, -1, d_model)  # [batch_size, max_pred, d_model]
        h_masked = torch.gather(output, 1, masked_pos)  # masking position [batch_size, max_pred, d_model]
        h_masked = self.activ2(self.linear(h_masked))  # [batch_size, max_pred, d_model]
        logits_lm = self.fc2(h_masked)  # [batch_size, max_pred, vocab_size]
        return logits_lm, logits_clsf
