## Project 3 报告

### 作业要求

1、基于RNN实现文本分类任务，数据使用搜狐新闻数据(SogouCS, 网址：http://www.sogou.com/labs/resource/cs.php)。任务重点在于搭建并训练RNN网络来提取特征，最后通过一个全连接层实现分类目标。
可以参考https://zhuanlan.zhihu.com/p/26729228

2、基于CIFAR-10数据集（https://www.cs.toronto.edu/~kriz/cifar.html）使用CNN完成图像分类任务。
3、基于MNIST数据集（http://yann.lecun.com/exdb/mnist/）使用GAN实现手写图像生成的任务。



### 1. RNN实现文本分类

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

![1](C:\Users\耿岱琳\Desktop\git人工智能导论作业\第三次作业\image\1.PNG)