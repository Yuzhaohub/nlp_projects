"""
  @Date    : 2021/9/24 10:36
  @Author  : fisher
  @File    : bert_base
  @Software: PyCharm
  @desc: bert预训练模型相关配置文件
"""

import os
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
# BASE_DIR = BASE_DIR + '/albert_model'

bert_config = {

    'bert_dir': BASE_DIR + '/bert_base_chinese',
    'bert_config_path': BASE_DIR + '/bert_base_chinese/config.json',
    'bert_vocab_path': BASE_DIR + '/bert_base_chinese/vocab.txt'
}


if __name__ == '__main__':
    print(BASE_DIR)
    print(bert_config['bert_dir'])
    print(bert_config['bert_config_path'])


