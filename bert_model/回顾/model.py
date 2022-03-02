# -*- coding: utf-8 -*-
# @Time    : 2021/11/18 10:59
# @Author  : Fisher
# @File    : model.py
# @Software: PyCharm
# @Desc    : Bert模型

"""
Bert模型Encoder部分：
    1、嵌入层
    2、多头注意力 + 残差链接
    3、前馈神经网络 + 残差链接
    4、前馈神经网络预测
注意：
    1、残差链接
    2、多头注意力机制
"""
import numpy as np
import torch
import torch.nn as nn
import math


class Config:
    """ 配置参数 """

    def __init__(self):
        self.vocab_size = 200
        self.d_model = 100
        self.n_segments = 2
        self.d_k = 64
        self.d_v = 64
        self.n_heads = 12
        self.d_ff = 768 * 4
        self.maxlen = 30
        self.n_layers = 12
        self.dropout = 0.5


def gelu(x):
    """激活函数：
    0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
    x: [batch_size, seq_len, d_model]
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def get_attn_pad_mask(seq_q, seq_k):
    """Padding Mask操作

    Parameter：
        seq_q: [batch_size, seq_len]
        seq_k: [batch_size, seq_len]

    Return:
        [batch_size, seq_len, seq_len]
    """
    batch_size, seq_len = seq_q.size()
    pad_attn_mask = seq_q.data.eq(0).unsqueeze(1)  # [batch_size, 1, seq_len]
    return pad_attn_mask.expand(batch_size, seq_len, seq_len)


class Embedding(nn.Module):
    """ 嵌入层：词嵌入层 + 位置编码 """

    def __init__(self, config):
        super(Embedding, self).__init__()
        self.tok_embed = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_embed = nn.Embedding(config.maxlen, config.d_model)
        self.seg_embed = nn.Embedding(config.n_segments, config.d_model)
        self.norm = nn.LayerNorm(config.d_model)

    def forward(self, x, seg):
        """嵌入层：词向量 + 位置编码 + 句子从属编码
        x: [batch_size, seq_len]
        seg: [batch_size, seq_len]
        """
        seq_len = x.size(1)
        pos = torch.arange(seq_len, dtype=torch.long)
        pos = pos.unsqueeze(0).expand_as(x)
        embedded = self.tok_embed(x) + self.pos_embed(pos) + self.seg_embed(seg)
        return self.norm(embedded)


class ScaledDotProductAttention(nn.Module):
    """ 注意力机制运算模块 """

    def __init__(self, config):
        self.config = config

        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V, attn_mask):
        """
        Parameter:
            Q: [batch_size, n_heads, seq_len, d_k]
            V: [batch_size, n_heads, seq_len, d_k]
            K: [batch_size, n_heads, seq_len, d_k]
            attn_mask: [batch_size, n_heads, seq_len, seq_len]
        """

        scores = Q.matmul(K.tranpose(-1, -2)) / np.sqrt(self.config.d_k)
        scores.mask_fill_(attn_mask, -1e9)  # 在softmax之前对权重系数进行Pad_Mask操作
        attn = nn.Softmax(dim=-1)(scores)  # attn: [batch_size, n_heads, seq_len, seq_len]

        context = attn.matmul(V)  # [batch_size, n_heads, seq_len, d_k]
        return context


class MultiHeadAttention(nn.Module):
    """ 多头注意力机制: Q、K、V矩阵 """

    def __init__(self, config):
        super(MultiHeadAttention, self).__init__()
        self.config = config
        self.scaledotattn = ScaledDotProductAttention(config)

        self.W_Q = nn.Linear(config.d_model, config.d_k * config.n_heads)
        self.W_K = nn.Linear(config.d_model, config.d_k * config.n_heads)
        self.W_V = nn.Linear(config.d_model, config.d_k * config.n_heads)

    def forward(self, Q, K, V, attn_mask):
        """计算公式： QK_T*V / d_k_(1/2)
        Parameter:
            Q: [batch_size, seq_len, d_model]  --> [batch_size, n_heads, seq_len, d_k]
            V: [batch_size, seq_len, d_model]  --> [batch_size, n_heads, seq_len, d_k]
            K: [batch_size, seq_len, d_model]  --> [batch_size, n_heads, seq_len, d_k]
            attn_mask: [batch_size, seq_len, seq_len],padding mask在softmax之前
        """
        residual, batch_size = Q, Q.size(0)
        q_s = self.W_Q(Q).view(batch_size, -1, self.config.n_heads, self.config.d_k).transpose(1, 2)
        k_s = self.W_K(K).view(batch_size, -1, self.config.n_heads, self.config.d_k).transpose(1, 2)
        v_s = self.W_V(V).view(batch_size, -1, self.config.n_heads, self.config.d_k).transpose(1, 2)

        attn_mask = attn_mask.unsqueez(1).repeat(1, self.config.n_heads, 1, 1)
        context = self.scaledotattn(q_s, k_s, v_s, attn_mask)  # [batch_size, n_heads, seq_len, d_k]
        context = context.transpose(1, 2).view(batch_size, -1, self.config.n_heads * self.config.d_k)

        output = nn.Linear(self.config.n_heads * self.config.d_k, self.config.d_model)(context)
        return output


class PoswiseFeedForwardNet(nn.Module):
    """ 前馈神经网络 """

    def __init__(self, config):
        super(PoswiseFeedForwardNet, self).__init__()
        self.fc1 = nn.Linear(config.d_model, config.d_ff)
        self.fc2 = nn.Linear(config.d_ff, config.d_model)

    def forward(self, x):
        """流程：经过词嵌入层、注意力层、层归一化层、残差连接层

        Parameter:
            x: [batch_size, seq_len, d_model]，经过多头注意力机制处理后的特征
        """
        return self.fc2(gelu(self.fc1(x)))


class EncoderLayer(nn.Module):
    """ Transformer编码层:encoder block """

    def __init__(self, config):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention(config)
        self.pos_ffn = PoswiseFeedForwardNet(config)

    def forward(self, enc_inputs, enc_self_attn_mask):
        """ 流程：多头注意力机制、残差链接 + 前馈神经网络、残差链接

        Parameter：
            enc_inputs: [batch_size, seq_len, d_model]
            enc_self_attn_mask: [batch_size, seq_len, seq_len]
        """
        enc_outputs = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs, enc_self_attn_mask)
        enc_outputs = self.pos_ffn(enc_outputs)  # [batch_size, seq_len, d_model]
        return enc_outputs


class BERT(nn.Module):
    """ Bert模型：12层Encode编码层 """

    def __init__(self, config):
        super(BERT, self).__init__()
        self.config = config
        self.embedding = Embedding(config)

        self.layers = nn.ModuleList(
            [EncoderLayer(config) for _ in range(config.n_layers)]
        )
        self.fc = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            nn.Dropout(config.dropout),
            nn.Tanh()
        )

        # 用于判断是否为一个句子
        self.classifier = nn.Linear(config.d_model, 2)
        self.linear = nn.Linear(config.d_model, config.d_model)
        self.active2 = gelu

        # 该部分共享参数
        embed_weight = self.embedding.tok_embed.weight  # nn.Embedding(vocab_size, d_model): [vocab_size, d_model]
        self.fc2 = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.fc2.weight = embed_weight  # 用于反向检索：embedding将vocab映射为d_model, 该部分将d_model映射为vocab_size

    def forward(self, input_ids, segment_ids, masked_pos):
        """流程：嵌入层、注意力、残差链接、层归一化

        Parameter:
            input_ids: [batch_size, seq_len]
            segment_ids: [batch_size, seq_len]
            masked_pos: [batch_size, max_pred],MASK处理的位置
        """
        output = self.embedding(input_ids, segment_ids)  # [batch_size, seq_len, d_model]
        enc_self_attn_mask = get_attn_pad_mask(input_ids, input_ids)  # [batch_size, seq_len, seq_len]
        for layer in self.layers:
            output = layer(output, enc_self_attn_mask)  # [batch_size, seq_len, d_model]

        h_pooled = self.fc(output[:, 0])  # 获取[CLS]特征向量表示，进行分类问题
        logits_clsf = self.classifier(h_pooled)  # [batch_size, 2] 句子分类用：[CLS]向量

        masked_pos = masked_pos[:, :, None].expand(-1, -1, self.config.d_model)
        h_masked = torch.gather(output, 1, masked_pos)  # 从原始向量找到MASK处理后的编码特征向量
        h_masked = self.active2(self.linear(h_masked))  # [batch_size, max_pred, vocab_size]
        logits_lm = self.fc2(h_masked)

        return logits_lm, logits_clsf
