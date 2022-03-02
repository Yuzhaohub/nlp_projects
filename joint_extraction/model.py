"""
  @Date    : 2021/10/18 15:12
  @Author  : fisher
  @File    : model
  @Software: PyCharm
  @desc: 模型构建：
        1、subject模型：随机抽取一个subject用于训练
        2、predicate|object模型，通过”半指针-半标注“
"""

"""
模型结构：
    1、subject预测网络
    2、object|predicate网络（随机抽取一个subject对应的）
"""

import torch
import torch.nn as nn


def seq_max_pool(x):
    """ 最大池化操作 """
    seq, mask = x
    seq = seq - (1 - mask) * 1e10  # mask: 表示掩码部分，1表示pad， 0表示非pad

    # torch.max(seq, dim = 1): seq: [bath_size, seq_len, embed_size]
    return torch.max(seq, 1)  # [batch_size, embed_size]  ----> 针对seq_len, 相当于获取embedding每个分量上的最大值


def seq_and_vec(x):
    """ 数据增强：原始向量+最大池化向量 """
    seq, vec = x   # seq: [batch_size, seq_len, embed_size] vec: [batch_size, embed_size]
    vec = vec.unsqueeze(dim = 1)

    # [batch_size, 1, embed_size] + [batch_size, seq_len, 1] --> [batch_size, seq_len, embed_size]
    vec = torch.zeros_like(seq[:, :, :1]) + vec

    return torch.cat([seq, vec], dim = 2)


def seq_gather(x):
    """ 根据索引获取特征向量: [batch_size, embed_size] """
    seq, ids = x  # seq: [batch_size, seq_len, embed_size] ids: [batch_size, 1]
    ids = ids.squeeze(-1)

    res = []
    for i in range(len(ids)):
        res.append(seq[i, ids[i], :])  # [embed_size]

    return torch.stack(res, dim = 0)  # [batch_size, embed_size]


class S_Model(nn.Module):
    """ Subject实体抽取网络
    网络结构：
        特征层：两层LSTM网络、CNN网路【垂直抽取】
        全连接层：双层网络定位索引
    细节：
        mask机制、特征融合（数据增加部分）
    """

    def __init__(self, vocab_size, embed_size, device):
        super(S_Model, self).__init__()

        self.device = device
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.dropout1 = nn.Sequential(
            nn.Dropout(0.25)
        )

        # 特征抽取层：两层垂直lstm层 + 一层cnn层
        self.lstm1 = nn.LSTM(
            input_size = embed_size,
            hidden_size = embed_size // 2,
            num_layers = 1,
            batch_first = True,
            bidirectional = True
        )

        self.lstm2 = nn.LSTM(
            input_size = embed_size,
            hidden_size = embed_size // 2,
            num_layers = 1,
            batch_first = True,
            bidirectional = True
        )

        self.conv = nn.Sequential(
            nn.Conv1d(
                in_channels = embed_size * 2,
                out_channels = embed_size,
                kernel_size = 3,
                padding = 1,
                stride = 1
            )
        )

        self.fc_ps1 = nn.Linear(embed_size, 1)
        self.fc_ps2 = nn.Linear(embed_size, 1)

    def forward(self, x):
        """
        subject判别网络：
        x: [batch_size, seq_len]
        mask: [batch_size, seq_len, 1]
        """
        mask = torch.gt(torch.unsqueeze(x, 2), 0).type(torch.FloatTensor).to(self.device)  # torch.gt()|torch.eq()
        mask.requires_grad = False

        # embedding的mask机制处理：词向量与mask矩阵对应位置相乘
        embeded = self.embed(x)  # [batch_size, seq_len, embed_size]

        t = embeded
        t = self.dropout1(t)

        t = t.mul(mask)  # mul对应位置相乘：遮蔽padding做词向量的影响

        t, (h_n, c_n) = self.lstm1(
            t)  # t: [batch_size, seq_len, bidrectional * hidden_size], c: [batch_size, num_layers * num_directions, hidden_size]
        t, (h_n, c_n) = self.lstm2(t)

        # 数据增强操作：最大池化操作：seq - (1 - mask) * 1e10
        t_max, t_max_index = seq_max_pool([t, mask])  # 最大池化操作：针对seq_len维度

        # 数据增加：torch.cat([t, t_max])，将最大池化的向量拼接到特征向量
        h = seq_and_vec([t, t_max])  # [batch_size, seq_len, embed_size * 2]

        # 卷积操作：input: [batch_size, in_channels, embed_size] kernel: [in_channels, out_channels, kernel_size]
        # output：[batch_size, out_channels, cons_size]  针对[in_channels, embed_size] 固定in_channels，在宽的维度上滚动
        h = h.permute(0, 2, 1)  # [batch_size, embed_size * 2, seq_len]

        h = self.conv(h)  # [batch_size, out_channels, seq_len]
        h = h.permute(0, 2, 1)

        ps1 = self.fc_ps1(h)
        ps2 = self.fc_ps2(h)

        # ps1: subject起始位置 ps2: subject终止位置 t: 特征向量(lstm模型) t_max: 最大池化特征向量（lstm模型）a
        return [ps1, ps2, t, t_max, mask]


class PO_Model(nn.Module):
    """ predicate|object实体抽取
    模型：随机抽取subject对其对应的object|predicate预测
        特征向量：
            1、lstm模型抽取的特征向量：[batch_size, seq_len, embed_size]
            2、最大池化特征向量
            3、抽取的subject对象词向量
        特征融合：
            卷积神经网络
        全连接层：
            多目标预测：[batch_size, seq_len, output_size]
    """

    def __init__(self, embed_size, num_classes):
        super(PO_Model, self).__init__()
        self.conv = nn.Conv1d(
            in_channels = embed_size * 4,
            out_channels = embed_size,
            kernel_size = 3,
            stride = 1,
            padding = 1
        )

        self.fc_po1 = nn.Linear(embed_size, num_classes + 1)
        self.fc_po2 = nn.Linear(embed_size, num_classes + 1)

    def forward(self, t, t_max, k1, k2):
        """
        特征融合：
            1、lstm抽取词向量t: [batch_size, seq_len, embed_size]
            2、最大池化特征向量t_max: [batch_size, embed_size]
            3、subject对象词向量k1, k2: [batch_size, 1]
        """
        # 特征拼接、融合：加入特征首尾特征
        k1 = seq_gather([t, k1])  # [batch_size, embed_size]  加入实体首部特征
        k2 = seq_gather([t, k2])  # [batch_size, embed_size]  加入实体尾部特征

        k = torch.cat([k1, k2], dim = 1)

        # 数据增强：[t, t_max]特征向量与最大池化向量
        h = seq_and_vec([t, t_max])  # [batch_size, seq_len, embed_size * 2]

        # k: [batch_size, embed_size * 2] 会将k特征向量广播后，重新拼接、融合，针对seq_len进行增强
        h = seq_and_vec([h, k])  # [batch_size, seq_len, embed_size * 4]，加入实体首尾特征

        h = h.permute(0, 2, 1)
        h = self.conv(h)  # [batch_size, out_channels, conv_size]

        h = h.permute(0, 2, 1)  # [batch_size, out_channels, conv_size]
        po1 = self.fc_po1(h)  # [batch_size, seq_len, num_classes + 1]
        po2 = self.fc_po2(h)  # [batch_size, seq_len, num_classes + 1]

        return [po1, po2]
