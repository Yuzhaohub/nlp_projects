#### Transformer模型细节

###### Transformer模型结构：

* Encoder（编码层）

  * Embedding（编码层）：

    * Position Encoding（位置嵌入）：**因为Transformer不采用RNN的结构，而是使用全局信息，不能利用单词的顺序信息，而这部分信息对于NLP来说非常重要**

      ![image-20220104110005363](C:\Users\Lenovo\AppData\Roaming\Typora\typora-user-images\image-20220104110005363.png)

    * Embedding（词向量嵌入）

    * 句子归属嵌入（0,1区分第几个句子）

  * Multi-Head Attention（多头注意力层）：**Q（查询）、K（键值）、V（值）**

    * self-Attention的输入（单词的表示向量、或者上一个Encoder block的输出），则可以使用线性变化矩阵**WQ、Wk、WV**计算得到**Q、K、V**，**X、Q、K、V的每一行都表示一个单词**

      ![image-20220104112537135](C:\Users\Lenovo\AppData\Roaming\Typora\typora-user-images\image-20220104112537135.png)

    * **注意：1、计算矩阵Q、K每一行向量的内积，为了防止内积过大，因此除以dk的平方根**

  * Add & Norm（残差连接、层归一化）：**1、Add表示残差连接：用于防止网络退化；2、Norm表示Layer Normalization：用于对每一层的激活值进行归一化**

    * 输入：**多头注意力输入**或者**全连接前馈层**
    * **Add：通常用于解决多层网络训练的问题，可以让网络只关注当前差异的部分**
    * **Norm：Layer Normalization，通常用于RNN结构，Layer Normalization会将每一层神经元的输入都转换成均值方差一样的，这样可以加快收敛**

  * Feed Forward：**是一个两层的全接连层，第一层的激活函数为ReLu，第二层不使用激活函数，注意：Feed Forward最终得到的输出矩阵的维度与X（词嵌入层）一致**

* Decoder（解码层）：

  * Embedding（编码层）与Encoder的编码层相同

  * **Mask Multi-Head Attention（掩码多头注意力机制）：**通过Masked操作可以防止第i个单词之后的信息。**注意：Mask操作是在Self-Attention的Softmax之前操作**

    * **理解：V，每一行表示一个词的词向量，通过下三角矩阵进行转换**

    ![image-20220104134215019](C:\Users\Lenovo\AppData\Roaming\Typora\typora-user-images\image-20220104134215019.png)

  * Decoder block第二个Multi-Head Attention变化不大，**主要区别在于其中Self-Attention的K，V矩阵不是使用上一个Decoder block的输出计算，而是使用Encoder的编码信息矩阵C计算的**
    * 根据Encoder的输出C计算得到K，V，根据上一个Decoder block的输出Z计算Q。
    * **好处：每一位单词都可以利用到Encoder所有单词的信息（这部分信息无需Mask）**

###### Bert面试版本：

* **1、Bert工作原理**
  * 与传统的RNN模型相比：Transformer在串行的基础上引入并行，加快运算速度。
  * 与传统的词向量Word2vec相比：Bert加入了上下文进行分析，不同的词在不同的语境中表达的意思不一样，让词之间有更好的解释性。
  * Self-Attention（自注意力机制）：在bert里不是人为的给权重而是计算机自动的基于上下文给权重，让计算机去注意哪些关键特征和信息。
  
* **2、Bert中的self-attention机制**
  * embedding，经过编码后得到词向量（包括position embedding）
  * 得到此前词语上下文的关系，可以当做加权。
  * 基于输入X（embedding或者是全连接层输出）构建三个矩，Queries、Keys、Values，分别查询当前词和其他词之间的关系（相当于计算相关系数），以及特征向量的表达。
  * **计算：Q与K的内积表示多匹配，内积越大，相关性越高，关系越近。得到的结果经过Softmax就是上下文的结果，Softmax就是将分值转换成概率，为了防止内积方差过大，剔除向量纬度造成的影响。**
  * **Self-Attention做的是矩阵乘法，就是一个句子的词全部一起算出来，这个就是并行加速。**
  
* **3、Q、K、V矩阵**
  
  * Q：查询的，词之间的关系
  * K：等被查的
  * V：实际的特征信息
  * Q*K和内积有关，相关程度。内积越大，相关性越高，关系越近。
  
* **4、Softmax激活函数：**
  
  * 分值转化：exp映射，归一化。先做exp再normarization。输出值转换为范围在[0,1]和为1的概率分布。
  * softmax后乘上对应的Value，一个词和这个句子中的所有词做这个操作最后加起来，得到最终分数。
  
* **5、Attention整体计算流程：**

  * 每一词的Q会跟每一个K计算得分
  * softmax后就得到整个加权结果
  * 此时每一个词看的不只是它前面的序列而是整个输入序列
  * 同一时间计算出所有词的表示结果

* **6、Multi-Head（多头注意力机制）**

  * 每个Self-Attention后有多头机制，用multi-head注意力机制综合多个特征，拼在一起进行降纬处理，**不同的注意力机制得到的特征向量表达是不一样的**，增强信息利用率和特征提取能力，以此提高模型性能。
  * 通过不同的head得到多个特征表达
  * 将所有特征拼接在一起
  * 可以通过一层全连接层进行降维

* **7、位置编码（Position Encoding）**

  * 为了让模型捕捉到单词的顺序信息，添加位置编码向量信息，位置编码向量不需要训练，通过构建规则产生。
  * 在偶数位置，使用正弦编码，在奇数位置，使用余弦编码。**编码维度与词向量维度一致**。最后把这个position encoding与embedding的值相加，作为输入送到下一层。

* **8、Layer Normalization（层归一化）**

  * 在Transformer中，每一个子层（self-attention， ffnn）之后都会接一个残差模块，并且有一个Layer Normalization。
  * **Layer Normalization的目的：把输入转化成均值为0，方差为1的数据，不希望输入数据落在激活函数的饱和区**

* **9、Layer Normalization和Batch Normalization的区别**

  ![image-20220104150417765](C:\Users\Lenovo\AppData\Roaming\Typora\typora-user-images\image-20220104150417765.png)

  * Batch Normalization的处理对象是对一批样本，Layer Normalization的处理对象是单个样本。
  * **Batch Normalization是这批样本的同一维度特征做归一化，Layer Normalization是对这单个样本的所有维度特征做归一化**
  * **Batch Norm就是通过对batch size这个维度归一化让分布稳定下来。Layer Norm则是通过对Hidden size这个维度归一化**，经过归一化再输入激活函数，得到的值大部分落入非线性函数的先行区，导数远离导数饱和区，避免了梯度消失，从而加速训练收敛过程。

* **10、Decoder相关：Mask机制**

  * mask表示掩码，它对某些值进行掩盖，使其在参数更新时不产生效果。**Transformer模型里面涉及两种mask，分别是Padding Mask和Sequence Mask**


* **11、静态mask**
  * 输入时，随机遮蔽或替换一句话里面任意字或词，然后让模型通过上下文的理解预测那一个被遮蔽或替换的部分，之后做的时候只计算被遮蔽部分的。**随机将一句话中15%的替换成一下内容：**
    * 80%的几率替换成"[MASK]"
    * 10%的利率替换成任意一个其他的字符
    * 10%的几率原封不动
* **12、动态mask**
  * 将原始数据复制n份，每份都进行随机的静态mask，从而每份数据的mask结果都不太一样。
  * 在每一个epoch的mask策略都不同。









