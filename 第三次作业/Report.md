# Project 3 报告

### 作业要求

1、基于RNN实现文本分类任务，数据使用搜狐新闻数据(SogouCS, 网址：http://www.sogou.com/labs/resource/cs.php)。任务重点在于搭建并训练RNN网络来提取特征，最后通过一个全连接层实现分类目标。
可以参考https://zhuanlan.zhihu.com/p/26729228

2、基于CIFAR-10数据集（https://www.cs.toronto.edu/~kriz/cifar.html）使用CNN完成图像分类任务。
3、基于MNIST数据集（http://yann.lecun.com/exdb/mnist/）使用GAN实现手写图像生成的任务。



## 1. RNN实现文本分类

#### 数据预处理

数据共4021条， 每条数据包括label和text

统计后发现，'news', 'sports'和'business'三类各占2989、1200、1051，其余几类新闻非常少，数据分布不均，所以只选择'news', 'sports'和'business'三类新闻进行训练。

统计每条新闻长度后，删除文本长度小于50的新闻。

因为模型只能处理数字，所以需要将label和text都映射为数字。label的转换很直接，将三类分别映射为1，2，3，然后进行one-hot转换。而对于新闻文本，由于数据量较小，我选择不用现有的词库，而是将新闻文本中的所有字组成词库。没有使用词是因为数据较少，很可能出现词库中没有的字。

```python
#映射字
data['chars']=data['text'].apply(lambda x: re.findall('[\x80-\xff]{3}|[\w\W]',x))

#组成字库
all_chars = []
for c in data['chars']:
    all_chars.extend(c)
char_dict = pd.DataFrame(pd.Series(all_chars).value_counts())
char_dict['id'] = list(range(1, len(char_dict)+1))
```

最后截取每条新闻的前100个字，按照2：8的比例生成测试集和训练集

#### 建模

建模部分参考keras关于LSTM的官方文档。

```python
model = Sequential()
model.add(Embedding(len(char_dict)+1, 256)) 
model.add(Dropout(0.5))
model.add(Dense(y_train.shape[1]))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
```

训练结果如下：

在训练集中，accuracy不断增长，loss不断下降，效果还不错

![1](/Users/Blankchul/Desktop/IntroductinoToAI_homework-master/第三次作业/1_RNN/images/1.png)

![2](/Users/Blankchul/Desktop/IntroductinoToAI_homework-master/第三次作业/1_RNN/images/2.png)

但是在测试集中，loss比较大。![3](/Users/Blankchul/Desktop/IntroductinoToAI_homework-master/第三次作业/1_RNN/images/3.png)

应该是由于数据集比较小，出现了过拟合问题。





## 2.CNN实现图像分类

#### 数据预处理

CIFAR10的数据可以直接从touchvision中分为测试集合训练集导入，CIFAR官方的数据是numpy array，需要转化为tensor。

然后把rbg值转化到[0, 1]之间。

图片一共10个类别，classes = ('plane', 'car', 'bird', 'cat',  'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

#### 建立模型

基于LeNet建立模型，包含两个卷积层和三个全连接层，ReLU作为激活函数, MaxPool作为池化层。

Loss function 和 optimizer选择交叉熵和随机梯度下降。

#### 训练模型

为了简单只训练了5个epoch, 每2000张图片计算一次loss。结果如图，可以看到loss在不断下降。

![4](/Users/Blankchul/Desktop/IntroductinoToAI_homework-master/第三次作业/2_CNN/4.png)

#### 测试

随机选4张图片进行测试，结果基本准确

最后在1000张图片上预测，计算准确率。



## 3. GAN生成手写图像

这一部分的代码包括两个模型，GAN和DCGAN。

 DCGAN的结果以图片和gif的形式包含在3_GAN文件里，但是DCGAN训练需要的时间太久，并没有运行完所有50个 epoch，只运行了15个，但是已经可以看到明显的变清晰的趋势。