#### BERT模型压缩与优化

![image-20211231153315374](C:\Users\Lenovo\AppData\Roaming\Typora\typora-user-images\image-20211231153315374.png)

**Bert/Transformer模型的加速方法，体现了从建模依赖、数据依赖、硬件依赖的优化层次：**

* 模型结构精简化与知识蒸馏
* 模型量化（Quantization）
* 网络结构搜索（NAS：Network Architecture Search）
* 计算图优化（计算图等价变化）
* 推理优化引擎（Faster Transformer / Torch-TensorRT /  AICompiler）

##### 模型优化

###### Bert-base应用缺陷

* Bert-base核心12层transformer结构，参数量大，对显卡要求高，推断耗时
* text长度对推断时间是由影响，越长越耗时，且bert本身不适用文档级的任务
* 原始Bert-base对NLG（文本生成）任务不友好

###### Bert-base优化

* 模型压缩：蒸馏、剪枝、量化
* 结构重组，为适应下游任务基于bert和一些其他结构重组，如MASS、ERNIE-GEN做NLG、Triple bert等
* 精调与任务优化，如Albert丢弃NSP任务，bert-wwm 做词Mask，Erine做实体Mask，更大数据的精调出Roberta等

###### Bert压缩方案

* **Pruning剪枝**

  * **training剪枝 VS post-training剪枝**	
    * Post-training剪枝是指predict前直接剪枝，简单粗暴且无需再训练，但容易剪枝过度后（关键节点被剪掉）难复原
    * 在training时小步剪枝，即使剪掉重要的内容也可以在后续training中恢复的机会。
  * **链接权重：**、
    * 链接权重：权重剪枝类似mask为0，这种剪枝理论上并不会减少模型大小，但是可以在实现上通过稀疏矩阵做到。
    * 神经元：直接给神经元剪枝就像dropout，预训练截断整层神经元LayerDrop，finetune截断精调，这种精调物理上减少了模型大小，性能损失不大，性价比高
    * 权重矩阵：减少attention的multi-head的个数
  * **如何剪枝**
    * **saliency-based**，按重要性对模型各结构排序，剪掉最不重要的部分，**1、在loss中加入L1正则，有筛选参数作用，实质是一种自动对权重排序并剪枝方式**，**2、在BN层加入channel-wise scaling factor并对之加L1使之稀疏，然后裁剪scaling factor值小的部分对应的权重，ReLu激活函数使得神经元dead，从某个角度也是剪枝神经元**
    * **loss-based**，对loss越小的结构倾向于剪掉它，**OBD和OBS基于损失函数相对于权重的二阶导数来衡量网络中权重的重要程度来进行裁剪**，**避免二阶求导的改进方法：用归一化的目标函数相对于参数的导数绝对值来衡量重要程度**
    * **Feature reconstruction error**,它的work的原理是如果我剪掉这个结果没啥影响，即最小化特征输出的重建误差。

  