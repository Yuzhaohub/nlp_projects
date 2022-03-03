# -*- coding: utf-8 -*-
# @Time    : 2021/11/24 11:04
# @Author  : Fisher
# @File    : run.py
# @Software: PyCharm
# @Desc    : 模型训练与测试

from model import Discriminator, Generator
from config import Config
from data_process import load_data, show_images, deprocess_img
from torch.autograd import Variable
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import os
from torchvision.utils import save_image


def loade_model(m_path, isFull_train = False):
    """ 导入模型：增量训练、全量训练 """
    pass


def train(trainloader, config):
    """ 模型训练：判别器的输出是数据的类别的one-hot向量，而不是0/1

    结构：
        1、训练判别器D：
            1、计算真实样本的判别损失LossR
            2、计算合成样本（假样本）的判别损失LossF_0
            3、更新判别器
        2、训练生成器G：
            1、计算合成样本的判别损失LossF_1
            2、更新判别器
    """
    D_net = Discriminator(config).to(config.device)
    G_net = Generator(config).to(config.device)

    criterion = nn.BCELoss().to(config.device)
    d_optimizer = optim.Adam(D_net.parameters(), lr = 0.0003)
    g_optimizer = optim.Adam(G_net.parameters(), lr = 0.0003)

    index = 0
    for i in range(config.epochs):
        for (img, label) in trainloader:
            img = img.to(config.device)

            # 生成label的one-hot编码向量，且设置对应类别位置是1
            labels_onehot = np.zeros((img.shape[0], config.label_dim))  # [batch_size, label_dim]
            labels_onehot[np.arange(img.shape[0]), label.numpy()] = 1

            # 生成随机向量，也就是噪声，带有标签信息
            z = torch.randn(img.shape[0], config.noise_dim)  # [batch_size, noise_dim] 从正态分布生成随机噪声
            z = np.concatenate((z.numpy(), labels_onehot), axis = 1)  # [batch_size, noise_dim + label_dim]
            z = torch.from_numpy(z).float().to(config.device)

            # 真实数据标签和虚假数据标签
            real_label = torch.from_numpy(labels_onehot).float()  # [batch_size, label_dim] 真实label对应类别是1
            fake_label = torch.zeros(img.shape[0], config.label_dim)  # [batch_size, label_dim]

            # 计算真实图片的损失函数
            real_out = D_net(img)
            d_loss_real = criterion(real_out,
                                    real_label.to(config.device))  # real_out/real_label: [batch_size, label_dim]

            # 计算虚假图片的损失函数:
            fake_img = G_net(z)  # z:[batch_size, label_dim],随机噪声 + 类别one-hot
            # feke_out = D_net(fake_img.detach())    # 如果动态图存有梯度，只能被更新一次，更新过后的部分会被移除当前的动态图
            fake_out = D_net(fake_img)
            d_loss_fake = criterion(fake_out, fake_label.to(config.device))

            d_loss = d_loss_real + d_loss_fake

            # ===================================================================
            #                       反向传播理解（detach和retain_graph）
            # 1、detach（）:截断node反向传播的梯度流，将某个node变成变成不需要梯度的Varible时，梯度就不会从这个node往前面传播
            # loss.backward(): 计算梯度，optimizer.step(): 更新梯度
            # 流程：
            #   1、D判别器：判别真实图片_loss + G生成器图片_loss（不更新）
            #   1、G生成器：损失函数从D判别器流向G生成器，但是只更新G生成器
            # loss.backward(): 所有不需要的中间结果都会被删除。要么设置保留.backward(retain_graph=True),以进行不会删除中间结果的反向传播
            # ===================================================================

            d_optimizer.zero_grad()
            d_loss.backward(retain_graph = True)
            d_optimizer.step()

            for j in range(config.gepochs):
                # 共用一个噪声：
                fake_img = G_net(z)
                output = D_net(fake_img)

                # 此处用虚假图片的虚假标签去训练D（判别网络）
                g_loss = criterion(output, real_label.to(config.device))
                g_optimizer.zero_grad()  # 此时D判别器的损失会一直传播到生成器G
                g_loss.backward()
                g_optimizer.step()  # 并没有更新D判别器中的参数

            if (index + 1) % 500 == 0:
                try:
                    print('保存模型：{}/{}批次'.format(index + 1, config.epochs))
                    model_path = os.path.join(config.model_path, str(i) + '_g_net_model.pkl')
                    try:
                        torch.save(G_net.cpu().state_dict(), model_path)
                    except:
                        torch.save(G_net.state_dict(), model_path)
                except Exception as e:
                    print(e)
                print('模型校验')
                # TODO 数据与模型分别在GPU和CPU端
                test_labels = np.random.choice(config.label_dim, 10)
                predict(test_labels, G_net = G_net, s_name = str(i))
            index += 1
    try:
        print("判别器D模型保存......")
        model_path = os.path.join(config.model_path, 'd_net_model.pkl')
        torch.save(D_net.state_dict(), model_path)
    except Exception as e:
        print('模型保存失败......')
        print(e)


def predict(labels, G_net = None, m_path = None, s_name = None, is_gpu = False):
    """ 模型预测：指定条件生成对应的数字图片 """
    config = Config()

    if not isinstance(labels, np.ndarray):
        labels = np.asarray(labels)

    # TODO 追求线上的时效性：需要删除部分不必要的判断，以提高效率。
    if G_net is None:
        G_net = Generator(config)
        if m_path is None:
            if os.path.exists(config.model_path):
                m_name = os.listdir(config.model_path)[-1]
                m_path = os.path.join(config.model_path, m_name)
                G_net.load_state_dict(torch.load(m_path))
            else:
                raise FileNotFoundError('模型参数文件不存在')
        else:
            G_net.load_state_dict(torch.load(m_path))
    else:
        if m_path is not None:
            G_net.load_state_dict(torch.load(m_path))

    # 模型加载GPU需要耗费时间：因此预测部分在本地CPU预测
    G_net.eval()

    sample_len = len(labels)
    label_list = np.zeros((sample_len, config.label_dim))
    label_list[np.arange(sample_len), labels] = 1

    z = torch.randn(sample_len, config.noise_dim)
    z = np.concatenate((z.numpy(), label_list), 1)
    z = torch.from_numpy(z).float()

    if is_gpu:
        G_net.to(config.device)
        z.to(config.device)

    with torch.no_grad():
        real_img = G_net(z)
        print("标签：{} 对应得生成图片为：".format(labels))
        show_images(real_img.cpu())

        if s_name is not None:
            try:
                s_path = os.path.join(config.img_save_path, s_name + '.png')
                real_img = deprocess_img(real_img)
                save_image(real_img, s_path)
            except Exception as e:
                print("图片保存失败： {}".format(e))


if __name__ == '__main__':
    import time

    config = Config()
    m_path = r'D:\work\project\GAN\CGAN\source\3_g_net_model.pkl'
    G_net = Generator(config)
    G_net.load_state_dict(torch.load(m_path))

    start_time = time.time()
    labels = [7] * 100
    # predict(labels, G_net = G_net)
    predict(labels)
    cost_time = time.time() - start_time
    print('模型共花费：{}秒时间'.format(cost_time))
    # trainloader = load_data(config = config)
    # train(trainloader, config)
