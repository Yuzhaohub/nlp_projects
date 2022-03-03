# -*- coding: utf-8 -*-
# @Time    : 2021/11/3 15:48
# @Author  : Fisher
# @File    : train_eval.py
# @Software: PyCharm
# @Desc    : 模型训练与测试


"""
模型保存：
    1、bert参数
    2、分类器（是否需要分层保存）
模型提前截断：
最佳模型保存：
"""
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics
from utils import get_time_dif
from transformers import AdamW, get_linear_schedule_with_warmup
import warnings

warnings.filterwarnings('ignore')

# 初始化权重，默认xavier
def init_network(model, method='xavier', exclude='embedding', seed=123):
    """ 针对不同的层初始化权重向量
    exclude: 排除exclude，剩余权重初始化
    """
    for name, w in model.named_parameters():
        if exclude not in name:
            if len(w.size) > 2:
                continue
            if 'weight' in name:
                if method == 'xavier':
                    nn.init.xavier_normal_(w)
                elif method == 'kaiming':
                    nn.init.kaiming_normal_(w)
                else:
                    nn.init.normal_(w)
            elif 'bias' in name:
                nn.init.constant_(w, 0)
            else:
                pass


def evaluate(config, model, data_iter, test=False):
    """ 模型评估
    data_iter: (x, seq_len, mask), y
    """
    model.eval()
    loss_total = 0
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)

    index = 0
    with torch.no_grad():
        for texts, labels in data_iter:
            # 可以进行分批次预测，用list保存每个batch的数据
            outputs = model(texts)
            loss = F.cross_entropy(outputs, labels)
            loss_total += loss
            labels = labels.data.cpu().numpy()
            predict = torch.max(outputs.data, 1)[1].cpu().numpy()

            # TODO 数据维度不一致（划分样本批次时，数据不能完正划分），需要进行修改。
            if index == 77:
                print(texts)
            index += 1

            labels_all = np.append(labels_all, labels)
            predict_all = np.append(predict_all, predict)
    acc = metrics.accuracy_score(labels_all, predict_all)
    if test:
        report = metrics.classification_report(labels_all, predict_all, target_names=config.class_list, digits=4)
        confusion = metrics.confusion_matrix(labels_all, predict_all)
        return acc, loss_total / len(data_iter), report, confusion
    return acc, loss_total / len(data_iter)


def test(config, model, test_iter):
    """ 模型测试 """
    model.load_state_dict(torch.load(config.save_path))
    model.eval()
    start_time = time.time()
    test_acc, test_loss, test_report, test_confusion = evaluate(config, model, test_iter, test=True)
    msg = 'Test Loss: {0:>5.2},  Test Acc: {1:>6.2%}'
    print(msg.format(test_loss, test_acc))
    print("Precision, Recall and F1-Score...")
    print(test_report)
    print("Confusion Matrix...")
    print(test_confusion)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)


def train(config, model, train_iter, dev_iter, test_iter):
    """
    模型训练：
        1、学习率预热处理：
        2、最优模型保存（bert+分类器）
        3、模型提前截断：通过损失增减
    """
    start_time = time.time()
    model.train()
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']  # 正则化操作
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    # 1、模型学习率预热处理：避免模型前期出现的震荡
    # optimizer = torch.optim.Adam(model.parameters(),lr = config.learning_rate)
    optimizer = AdamW(optimizer_grouped_parameters, lr=config.learning_rate)
    total_steps = len(train_iter) * config.num_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0.05 * total_steps,
                                                num_training_steps=total_steps)
    total_batch = 0
    dev_best_loss = float('inf')
    last_improve = 0  # 记录上次验证集loss下降的batch数
    flag = False  # 记录是否很久没有效果提升

    for epoch in range(config.num_epochs):
        print('Epoch [{}/{}]'.format(epoch + 1, config.num_epochs))
        for i, (trains, labels) in enumerate(train_iter):
            outputs = model(trains)
            model.zero_grad()
            loss = F.cross_entropy(outputs, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()

            if total_batch % 100 == 0:
                true = labels.data.cpu()
                predic = torch.max(outputs.data, 1)[1].cpu()
                train_acc = metrics.accuracy_score(true, predic)

                # TODO 报错信息：
                dev_acc, dev_loss = evaluate(config, model, dev_iter)

                # last_improve：记录上次验证集loss下降的batch数， （total_batch - last_improve）
                if dev_loss < dev_best_loss:
                    dev_best_loss = dev_loss
                    # 模型保存：包含bert和分类器两部分
                    torch.save(model.state_dict(), config.save_path)
                    improve = '*'
                    last_improve = total_batch
                else:
                    improve = ''
                time_dif = get_time_dif(start_time)
                msg = 'Iter: {0:>6},  Train Loss: {1:>5.2},  Train Acc: {2:>6.2%},  ' \
                      'Val Loss: {3:>5.2},  Val Acc: {4:>6.2%},  Time: {5} {6}'
                print(msg.format(total_batch, loss.item(), train_acc, dev_loss, dev_acc, time_dif, improve))
                model.train()
            total_batch += 1
            if total_batch - last_improve > config.require_improvement:
                print("No optimization for a long time, auto-stopping...")
                flag = True
                break
        if flag:
            break

    # 模型预测：针对测试集数据。
    test(config, model, test_iter)


if __name__ == '__main__':
    from bert_sentiment.utils import build_dataset, build_iterator
    from bert_sentiment.model.bert_cnn import Config, Model

    config = Config()
    train_data, dev_data, test_data = build_dataset(config=config)
    train_iter, dev_iter, test_iter = build_iterator(train_data, config), build_iterator(dev_data,
                                                                                         config), build_iterator(
        test_data,
        config)

    model = Model(config=config)
    train(config, model, train_iter, dev_iter, test_iter)
