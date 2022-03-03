#### Bert模型Fine-Tuning
##### transfer learning
* 特征提取器：将预训练模型当做固定的模型，进行特征提取；然后构造分类器进行分类
* 微调预训练模型：可以将整个模型都进行参数更新，或者冻结前半部分网络，对后半段网络进行参数更新，因为前半段网络提取的是通用的
低级特征，后半段提取的是与数据集有关的特征
  
##### How to fine-tuning
* 新数据集小，与原始数据集类似：只训练最后的分类器
* 新数据集大，与原始数据集类似：微调整个网络或前半段网络
* 新数据集小，与原始数据集不类似：都不太适合，但可以试试对网络某一层的激活函数进行分类
* 新数据集大，与原始数据集不类似：重新训练整个模型，但是使用预训练的参数作为初始化

##### 注意事项
* 数据维度需要注意匹配，尤其fc层
* 微调模型的时候，对于预训练模型的学习率要较小，对新的分类层的学习率要较大，因为一般不破坏预训练模型的参数
        
        class Model(nn.Module):
            def __init__(self):
                super(Model,self).__init__()
                self.features = nn.Conv2d(in_channels=3, out_channels=64)
                self.classifiter = nn.Conv2d(in_channels=64, out_channels=128)
            def forward(x):
                relu = nn.ReLU()
                x = self.classifiter(self.features(x))
                return x
        
        lr = 1
        net = Model()
        optimizer = torch.optim.SGD([
            {'params': net.features.weight, 'lr': 0.5},
            {'params': net.features.bias, 'lr': 0.3},
            {'params': net.classifiter.weight, 'lr': 0.5},
            {'params': net.classifiter.weight}
        ], lr = 0.5)


##### Transformers学习器
warmup：是针对学习率learning rate优化的一种策略，
主要过程：
  1、在预热期间，学习率从0线性（也可非线性）增加到优化器中的初始预设lr
  2、之后使其学习率从优化器中的初始lr线性降低到0

理解：刚开始训练时，模型的权重（weights）是随机初始化的，此时若选择一个较大的学习率，可能带来模型的不稳定（震荡），选择Warmup预热学习率的方式，
可以使得开始训练的几个epochs或者一些step内学习率较小，在预热的小学习率下，模型可以慢慢趋于稳定，等模型相对稳定后再选择预先设置的学习率进行训练，
使得模型收敛速度变得更快，模型效果更加。
















