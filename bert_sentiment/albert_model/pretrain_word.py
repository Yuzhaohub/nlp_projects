"""
  @Date    : 2021/9/24 10:36
  @Author  : fisher
  @File    : pretrain_word
  @Software: PyCharm
  @desc: 词向量训练demo
"""

import numpy as np
import torch
from transformers import BertTokenizer, BertConfig, BertForMaskedLM, BertForNextSentencePrediction
from transformers import BertModel
import os

model_path = r'C:\Users\Lenovo\OneDrive\桌面\project\bert_sentiment\albert_model\bert_base_chinese'
vocab = 'vocab.txt'


# 导入分词器
tokenizer = BertTokenizer.from_pretrained(os.path.join(model_path, vocab))
model_config = BertConfig.from_pretrained(model_path + r'\config.json')

model_config.output_hidden_states = True
model_config.output_attentions = True
bert = BertModel.from_pretrained(model_path, config=model_config)

sen_code = tokenizer.encode_plus('这个故事没有终点', "正如星空没有彼岸")

"""
input_ids: 单词在词典中的编码
token_type_ids: 区分两个句子的编码（上句全为0，下句全为1）
attention_mask: 指定对哪些词进行self—attention操作
"""

print(tokenizer.convert_ids_to_tokens(sen_code['input_ids']))

tokens_tensor = torch.tensor([sen_code['input_ids']])
segments_tensors = torch.tensor([sen_code['token_type_ids']])

bert.eval()
with torch.no_grad():
    """
    outputs: 
        1、sequence_output： 输出序列  [batch_size, seq_len, hidden_size]
        2、pooled_output：对输出序列进行pool操作的结果 [batch_size, hidden_size]
        3、hidden_states：隐含层状态（包括Embedding层）(embedding + output of each layer)
        4、attentions：注意力层
    """
    outputs = bert(tokens_tensor, token_type_ids=segments_tensors)
    encoded_layers = outputs
    print(encoded_layers[0].shape, encoded_layers[1].shape,
          encoded_layers[2][0].shape, encoded_layers[3][0].shape)
