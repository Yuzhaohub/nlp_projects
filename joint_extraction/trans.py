"""
  @Date    : 2021/10/18 11:20
  @Author  : fisher
  @File    : trans
  @Software: PyCharm
  @desc: 数据转换：
            1、predicate关系集合整理: predicate2index、index2predicate
            2、训练集、测试集：[{'text': '', 'spo_list': [[subject, predicate, object], [], []}]
            3、字典表：word2index、index2word
"""

import json
from tqdm import tqdm
import codecs
import os

mkfiles = r'data/trans_data'
if not os.path.exists(mkfiles):
    os.makedirs(mkfiles)

# 构建关系集合：predicate2word、word2predicate
all_50_shcemas = set()
with codecs.open(r'data/all_50_schemas', encoding = 'utf-8') as f:
    for l in tqdm(f):
        a = json.loads(l)  # 将其解析为json
        all_50_shcemas.add(a['predicate'])

predicate2index = {v: i for i, v in enumerate(all_50_shcemas)}
index2predicate = {i: v for i, v in enumerate(all_50_shcemas)}

schemas_savefile = os.path.join(mkfiles, 'all_50_schmas_me.json')
with codecs.open(schemas_savefile, 'w', encoding = 'utf-8') as f:
    json.dump([predicate2index, index2predicate], f, indent = 4, ensure_ascii = False)

"""
训练集与测试集数据转换：
    格式： [{'text': '', 'spo_list': [[subject1, predicate1, object1], [subject2, predicate2, object2]]}]
"""

# 训练集
train_data = []
chartDict = {}
with codecs.open(r'data/ner_data/train_data.json', encoding = 'utf-8') as f:
    for l in tqdm(f):
        a = json.loads(l)
        spo_list = []
        for spo in a['spo_list']:
            spo_list.append([spo['subject'], spo['predicate'], spo['object']])
        train_data.append({'text': a['text'], 'spo_list': spo_list})
        for c in a['text']:
            chartDict[c] = chartDict.get(c, 0) + 1

train_data_savefile = os.path.join(mkfiles, 'train_json_me.json')
with codecs.open(train_data_savefile, 'w', encoding = 'utf-8') as f:
    json.dump(train_data, f, indent = 4, ensure_ascii = False)

# 测试集
dev_train = []
with codecs.open(r'data/ner_data/dev_data.json', encoding = 'utf-8') as f:
    for l in tqdm(f):
        a = json.loads(l)
        spo_list = []
        for spo in a['spo_list']:
            spo_list.append([spo['subject'], spo['predicate'], spo['object']])
        dev_train.append({'text': a['text'], 'spo_list': spo_list})
        for c in a['text']:
            chartDict[c] = chartDict.get(c, 0) + 1

dev_data_savefile = os.path.join(mkfiles, 'dev_json_me.json')
with codecs.open(dev_data_savefile, 'w', encoding = 'utf-8') as f:
    json.dump(dev_train, f, indent = 4, ensure_ascii = False)

# 保存字典
min_cnt = 2
chart_sort = dict(sorted(chartDict.items(), key = lambda row: row[1], reverse = True))
chart_sort_limit = {k: v for k, v in chartDict.items() if v >= min_cnt}
index2word = {i + 2: k for i, k in enumerate(chart_sort_limit.keys())}
word2index = {k: i for i, k in index2word.items()}

chartsfile = os.path.join(mkfiles, 'all_chars_me.json')
with codecs.open(chartsfile, 'w', encoding = 'utf-8') as f:
    json.dump([word2index, index2word], f, indent = 4, ensure_ascii = False)
