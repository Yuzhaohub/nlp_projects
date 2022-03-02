"""
  @Date    : 2021/10/19 14:12
  @Author  : fisher
  @File    : main
  @Software: PyCharm
  @desc: 实体、关系联合抽取入口
"""

"""
数据封装：
    1、数据清洗
    2、自定义数据集
    3、自定义加载器
"""

from mydataset import MyDataLoader, MydataSet, DataFormat
from model import S_Model, PO_Model
import yaml
import torch
from tqdm import tqdm
import numpy as np
import json


def load_configs(configsfile):
    """ 导入超参数的配置 """
    fopen = open(configsfile)
    configs = yaml.load(fopen, Loader = yaml.FullLoader)
    return configs


def extract_items(text_in, s_model, po_model, char2id, id2schemas, device, limit = 0.5):
    """ 单文本样本进行预测
    1、首先，s_model预测subject
    2、将所有的subject输入到po_model
    """
    R = []
    _s = [char2id.get(c, 1) for c in text_in]
    _s = np.array([_s])

    # s_m: ps_1, ps_2, t, t_max, mask ---->
    _k1, _k2, t, t_max, mask = s_model(torch.LongTensor(_s).to(device))
    _k1, _k2 = _k1[0, :, 0], _k2[0, :, 0]  # _k11: [batch_size, seq_len, 1] ----> [seq_len]，此时batch_size = 1
    _kk1s = []

    for i, _kk1 in enumerate(_k1):  # 遍历所有可能的抽取出来的subject：
        if _kk1 > limit:  # 输出为一个概率值，通过阈值判断是否为subject对象
            _subject = ''
            for j, _kk2 in enumerate(_k2[i:]):
                if _kk2 > limit:
                    _subject = text_in[i: i + j + 1]
                    break
            # 每次输入一个subjective对应的objective进行预测: [batch_size, seq_len] ==> [1, seq_len]
            if _subject:
                _k1, _k2 = torch.LongTensor([[i]]), torch.LongTensor(
                    [[i + j]])  # np.array([i]), np.array([i+j]) 加入实体首尾特征
                _o1, _o2 = po_model(t.cuda(), t_max.cuda(), _k1.cuda(), _k2.cuda())  # 使用到的特征：subjective层抽取的特征向量
                _o1, _o2 = _o1.cpu().data.numpy(), _o2.cpu().data.numpy()

                _o1, _o2 = np.argmax(_o1[0], 1), np.argmax(_o2[0], 1)

                for i, _oo1 in enumerate(_o1):
                    if _oo1 > 0:
                        for j, _oo2 in enumerate(_o2[i:]):
                            if _oo2 == _oo1:
                                _object = text_in[i: i + j + 1]
                                _predicate = id2schemas[_oo1]
                                # print((_subject, _predicate, _object))
                                R.append((_subject, _predicate, _object))
                                break
        _kk1s.append(_kk1.data.cpu().numpy())
    _kk1s = np.array(_kk1s)
    return list(set(R))


def predict(text_in):
    """ 批量预测 """
    pass


def evaluate(dev_data, s_model, po_model, char2id, id2schemas, device):
    A, B, C = 1e-10, 1e-10, 1e-10
    cnt = 0
    for d in tqdm(iter(dev_data)):
        R = set(extract_items(d['text'], s_model, po_model, char2id, id2schemas, device))
        T = set([tuple(i) for i in d['spo_list']])
        A += len(R & T)
        B += len(R)
        C += len(T)
        # if cnt % 1000 == 0:
        #     print('iter: %d f1: %.4f, precision: %.4f, recall: %.4f\n' % (cnt, 2 * A / (B + C), A / B, A / C))
        cnt += 1
        print('第{}条样本，抽取的实体SPO：{}'.format(cnt, R))
    return 2 * A / (B + C), A / B, A / C


def train(loader, dev_data, s_model, po_model, optimizer, epochs, loss, b_loss, char2id, id2schemas, device):
    """ 模型训练
    注意点：
        1、s_model与po_model参数联合更新
        2、s_model与po_model损失联合更新
    """

    best_f1 = 0
    best_epoch = 0
    best_precision = 0
    best_recall = 0

    for i in range(epochs):
        for step, loader_res in tqdm(iter(enumerate(loader))):
            # print(get_now_time())
            t_s = loader_res["T"].to(device)
            k1 = loader_res["K1"].to(device)
            k2 = loader_res["K2"].to(device)
            s1 = loader_res["S1"].to(device)
            s2 = loader_res["S2"].to(device)
            o1 = loader_res["O1"].to(device)  # o1、o2表示object
            o2 = loader_res["O2"].to(device)

            # s_m: subject抽取模型
            # s_m = s_model(len(char2id) + 2, CHAR_SIZE, HIDDEN_SIZE).cuda()
            ps_1, ps_2, t, t_max, mask = s_model(t_s)  # t_s.shape: [batch_size, seq_len]

            t, t_max, k1, k2 = t, t_max, k1, k2
            po_1, po_2 = po_model(t, t_max, k1, k2)

            ps_1 = ps_1
            ps_2 = ps_2
            po_1 = po_1
            po_2 = po_2

            s1 = torch.unsqueeze(s1, 2)  # [batch_size, seq_len] --> [batch_size, seq_len, 1]
            s2 = torch.unsqueeze(s2, 2)  # [batch_size, seq_len] --> [batch_size, seq_len, 1]

            # FloatTensor与LongTensor的区别：
            s1_loss = b_loss(ps_1, s1)  # 损失函数：nn.BCEWithLogitsLoss(): 输出为标量
            s1_loss = torch.sum(s1_loss.mul(mask)) / torch.sum(mask)  # mask处理机制：mask: [batch_size, seq_len, 1]
            s2_loss = b_loss(ps_2, s2)
            s2_loss = torch.sum(s2_loss.mul(mask)) / torch.sum(mask)

            po_1 = po_1.permute(0, 2,
                                1)  # [batch_size, seq_len, num_class + 1] --> [batch_size, num_class + 1, seq_len]
            po_2 = po_2.permute(0, 2,
                                1)  # [batch_size, seq_len, num_class + 1] --> [batch_size, num_class + 1, seq_len]

            o1_loss = loss(po_1, o1)  # 损失函数：nn.CrossEntropyLoss()
            o1_loss = torch.sum(o1_loss.mul(mask[:, :, 0])) / torch.sum(mask)
            o2_loss = loss(po_2, o2)
            o2_loss = torch.sum(o2_loss.mul(mask[:, :, 0])) / torch.sum(mask)

            loss_sum = 2.5 * (s1_loss + s2_loss) + (o1_loss + o2_loss)

            # if step % 500 == 0:
            # 	torch.save(s_m, 'models_real/s_'+str(step)+"epoch_"+str(i)+'.pkl')
            # 	torch.save(po_m, 'models_real/po_'+str(step)+"epoch_"+str(i)+'.pkl')

            optimizer.zero_grad()  # 梯度清零：每一批次梯度更新，而非模型参数归零。

            loss_sum.backward()
            optimizer.step()

        # 模型分块保存：s_m用于训练subject， po_m用于训练object、relation
        torch.save(s_model, 'models_real/s_' + str(i) + '.pkl')
        torch.save(po_model, 'models_real/po_' + str(i) + '.pkl')
        f1, precision, recall = evaluate(dev_data, s_model, po_model, char2id, id2schemas, device)

        print("epoch:", i, "loss:", loss_sum.data)

        if f1 >= best_f1:
            best_f1 = f1
            best_epoch = i
            best_precision = precision
            best_recall = recall

        print('f1: %.4f, precision: %.4f, recall: %.4f, bestf1: %.4f, bestepoch: %d \n ' % (
            f1, precision, recall, best_f1, best_epoch))

    # TODO 这部分信息可以写入配置文件中
    f = open(r'best_model.yaml', 'w')
    best_model = {
        'best_f1': best_f1,
        'best_epoch': best_epoch,
        'best_precision': best_precision,
        'best_recall': best_recall
    }
    yaml.dump(best_model, f)
    f.close()


def main():
    """ 实体、关系联合抽取入口 """

    # 超参数配置
    model_configs = r'model_config.yaml'

    configs = load_configs(model_configs)  # 在函数入口调用配置信息
    vocab_size = configs['vocab_size']
    embed_size = configs['embed_size']
    hidden_size = configs['hidden_size']
    num_classes = configs['num_classes']
    batch_size = configs['batch_size']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 模型整体加载到GPU
    s_model = S_Model(vocab_size, embed_size, device).to(device)
    po_model = PO_Model(embed_size, num_classes + 1).to(device)

    # 模型优化器
    params = list(s_model.parameters())
    params += list(po_model.parameters())
    optimizer = torch.optim.Adam(params, lr = 0.001)

    # 模型损失函数
    loss = torch.nn.CrossEntropyLoss().to(device)
    b_loss = torch.nn.BCEWithLogitsLoss().to(device)

    # 数据集封装
    print('====== 数据封装 ======')

    def loader(datapath):
        """ 数据封装 """
        data = json.load(open(datapath, encoding = 'utf-8'))
        mydataformat = DataFormat(data)
        res = mydataformat.pre_res()
        mydataset = MydataSet(*res)
        mydataloader = MyDataLoader(batch_size = batch_size)
        data_loader = mydataloader.loader(mydataset, mydataset.collate_fn)
        return data_loader, mydataformat

    dev_data_path = r'./data/trans_data/dev_json_me.json'
    dev_data = json.load(open(dev_data_path, encoding = 'utf-8'))

    train_data_path = r'./data/trans_data/train_json_me.json'
    train_data_loader, mydataset = loader(train_data_path)
    # dev_data_loader, _ = loader(dev_data_path)
    char2id = mydataset.word2id
    id2schemas = {v: k for k, v in mydataset.schemas2id.items()}

    # 模型训练与测试
    print('\n')
    print('==========================模型训练===============================')
    train(train_data_loader, dev_data, s_model, po_model,
          optimizer, 100, loss, b_loss, char2id, id2schemas, device)

    # 模型预测：还得重新加载：补充保存模型时，自动加载相关参数


if __name__ == '__main__':
    main()
