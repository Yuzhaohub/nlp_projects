# -*- coding: utf-8 -*-
# @Time    : 2021/11/4 10:51
# @Author  : Fisher
# @File    : train_model_block.py
# @Software: PyCharm
# @Desc    : 模型训练方面的技巧


"""
1、模型训练批次截断机制：依据损失（loss）的增减程度的批次
2、学习率预热处理：为防止模型在训练前期由于设置不合理的学习率而出现的震荡情况
3、最佳模型的筛选与保存：通过验证集的结果进行判别
"""

import torch
import torch.nn.functional as F
from sklearn import metrics

# 模型训练截断机制

def evaluate(config, model, dev_iter):
    return 0, 0

def model_truncate(model,train_iter, epochs, config, dev_iter, require_improvement):
    """ 模型截断处理
    total_batch: 移动指针
    last_improve：定位指针
    flag：结束状态标识
    dev_best_loss: 初始化
    """
    total_batch = 0
    dev_best_loss = float('inf')
    last_improve = 0  # 记录上次验证集loss下降的batch数（定位索引）
    flag = 0  # 记录是否很久没有效果提升，用于跳出循环，截断模型训练

    for epoch in epochs:
        for i, (trains, labels) in enumerate(train_iter):
            outputs = model(trains)
            model.zero_grad()
            # loss = F.cross_entropy(outputs, labels)
            # loss.backward()
            # scheduler.step()
            # optimizer.step()

            if total_batch % 100 == 0:
                true = labels.data.cpu()
                predic = torch.max(outputs.data, 1)[1].cpu()
                train_acc = metrics.accuracy_score(true, predic)
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
            total_batch += 1
            if total_batch - last_improve > require_improvement:
                flag = True
                break
        if flag:   # 跳出epochs的循环体
            break


def BertAdam(model, train_iter, epochs):
    """ 学习率预热处理：学习率从0逐步增加到预先设定的值，再逐步降为0 """
    from transformers import AdamW, get_linear_schedule_with_warmup

    param_optimizer = list(model.name_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']   # 正则化操作
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in p for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in p for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr = 0.01)
    total_steps = len(train_iter) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0.05 * total_steps,
                                                num_training_steps=total_steps)

    # 模型训练时，需要进行反向传播更新参数
    model.zero_grad()
    loss = torch.tensor([0.1])
    loss.backward()
    scheduler.step()
    optimizer.step()















