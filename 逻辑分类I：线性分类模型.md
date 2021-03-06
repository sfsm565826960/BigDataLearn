## 逻辑分类I：线性分类模型

### 引言
1. 输入方式：<br>
    非参数化模型：仅将输入的特征数值组合起来；<br>
    参数化模型：用带权重的数学表达式的方式组合输入特征数值；
    
2. 对事物的认知方式：<br>
    非参数化模型：如KNN，依靠近邻的数据集；<br>
    参数化模型：依靠参数矩阵；

3. 存储方式：<br>
    非参数化模型：如KNN，需要保留完整的训练集，占用空间大；<br>
    参数化模型：仅存储参数矩阵，占用空间小；

4. 时间复杂度：<br>
    非参数化模型：如KNN，训练快、预测慢；
    参数化模型：训练慢、预测快（更受欢迎）。

总结：线性模型把特征对分类结果的作用按权重比例加起来，用特征的权重矩阵表示模型对事物的认知。

------------------------

### 逻辑分类：预测
&emsp;&emsp;逻辑分类(Logistic Classification)*是一种线性模型，可以表示为y = w * x + b，其中
- w是训练得到的权重参数(weight)；
- x是样本特征数据；
- b是偏置(Bias)，可理解为一般情况下认为概率是多少。

> 注：一些资料也叫逻辑回归(Logistic Regression)，但本质是用作分类问题的。

&emsp;&emsp;逻辑分类模型预测一个样本分三步：
- 计算线性函数（y = w * x + b）；
- 从分数到概率的转换（Sigmoid 或 Softmax）；
- 从概率到标签的转换。


#### 第一步：线性函数
#### `打分函数`
```
def score(x, w, b):
    return np.dot(x, w) + b # np.dot为矩阵乘法函数
```
&emsp;&emsp;对于逻辑分类，训练模型的目的在于找到合适的`w`和`b`，让输出的预测值变得可靠。如下图，让直线变得尽可能接近所有样本点。
![image](https://timgsa.baidu.com/timg?image&quality=80&size=b9999_10000&sec=1509949179204&di=d9ccf913d79ccb295a78b286306f1009&imgtype=0&src=http%3A%2F%2Fimages2015.cnblogs.com%2Fblog%2F709432%2F201608%2F709432-20160819031052046-1651806152.png)


#### 第二步：将分数变成概率
#### `Sigmoid函数`
&emsp;&emsp;Sigmoid函数适用于只对一种类别进行判断的场景。<br>
- 设定函数阀值；
- 当Sigmoid大于阀值，则认为是；
- 否则认为不是。
```
def sigmoid(s):
    return 1. / (1 + np.exp(-s))
```
> 将输入的分数的范围映射在(0,1)之间，同时凸显大的分数的作用，抑制小的分数的作用。

#### `Softmax函数`
&emsp;&emsp;Softmax函数是Sigmoid函数的“多类别”版本，可以将输出值对应到多个类别标签，概率值最高的一项就是模型预测的标签。
```
def softmax(s):
    return np.exp(s) / np.sum(np.exp(s), axis = 0)
```
> 将输入的分数的范围映射在(0,1)之间，同时其中最大的分数并抑制远低于最大分数的其他数值。
-----------------------------

### 逻辑分类：评估
&emsp;&emsp;用更精确的数值衡量出 **预测样本** 和 **真实样本** 的表现差距。
#### `One-Hot编码`
&emsp;&emsp;因为预测结果是向量，因此需要将真实类别也转为向量。
- 真实类别向量维度与预测结果一致；
- 真实类别对应的概率值为1；
- 其他类别对应的概率值为0；
```
#例如
p = np.array([0.7, 0.2, 0.1]).reshape(-1,1) #预测结果向量
y = np.array([1, 0, 0]).reshape(-1,1) # 真实类别向量
```

#### `交叉熵`
&emsp;&emsp;衡量两个概率分布向量的差异程度。
```
## D(y,p) = yln(p) + (1-y)(1-ln(p))
def cross_entropy(y, p):
    #return np.sum(y * np.log(p) + (1 - y) * np.log(1 - p, axis=1)) 
    #上面语句语法错误，根据函数尝试改为下面语句
    return np.sum(y * np.log(p) + (1 - y) * (1 - np.log(p)), axis=1)
```
#### `Error差距`
```
Error  = -cross_entropy(y, p)
```
--------------------------

### 逻辑分类：训练
&emsp;&emsp;通过交叉熵，训练模型获得合适的`w`和`b`转换为寻找`w,b = min(∑Error)`
#### `损失函数`
```
# X是训练样本矩阵，w是权重，b是偏置向量，y是真实标签矩阵
def loss_func(X, w, b, y):
    s = score(X, w, b)
    y_p = softmax(s)
    return -np.mean(cross_entropy(y, y_p))
```
&emsp;&emsp;最原始的方法是暴力尝试所有w和b的权重参数组合，找到使`loss(w,b)`数值最小的`w`和`b`。

&emsp;&emsp;下一节（[逻辑分类II：线性分类模型](逻辑分类II：线性分类模型.md)），我们将介绍一些技巧让计算机更聪明地寻找恰当的参数组合。