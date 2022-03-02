#### RoBERTa模型
* 动态Mask（相当于十折交叉验证）：对数据进行了简单的增强，并且有一定的正则化效果
    
    * RoBERTa一开始就把预训练的数据复制10份，每一份都随机选择15%的Tokens进行Masking（同样的一句话有10种不同的mask方式）
    * 然后每份数据都训练N/10个epoch
    * 相当于在这N个epoch的训练中，每个序列的被mask的tokens是会变化的
    

理解：
* 对于在数据中随机选择15%的标记，其中80%被换成[MASK],10%不变，10%随机替换其他单词，原因是什么？

因为Bert用于下游任务微调时，[MASK]标记不会出现，它只出现在预训练任务中，会造成预训练和微调之间的不匹配，
微调不出现[MASK]这个标记，模型好像就没有了着力点、不知从何下手。
  