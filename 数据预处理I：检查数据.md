# 数据预处理I：检查数据
> 准备
```
import numpy as py
import pandas as pd
from sklearn import datasets
sk_iris = datasets.load_iris()
iris = pd.DataFrame(
    data= np.c_[sk_iris['data'], sk_iris['target']],
    columns= np.append(sk_iris.feature_names, 'target')
)
```
---------
> 查看样本集信息
```
>>>iris.info()
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 150 entries, 0 to 149
Data columns (total 5 columns):
sepal length (cm)    150 non-null float64
sepal width (cm)     150 non-null float64
petal length (cm)    150 non-null float64
petal width (cm)     150 non-null float64
target               150 non-null float64
dtypes: float64(5)
memory usage: 5.9 KB
```
---------
> 查看前N行数据 pd.DataFrame.head(n=5)
```
#返回前n行数据
>>>iris.head(2)
   sepal length (cm)  sepal width (cm)  petal length (cm)  petal width (cm)   target
0                5.1               3.5                1.4               0.2       0.0
1                4.9               3.0                1.4               0.2       0.0
```
----------------------
> 检查数据缺失 pd.DataFrame.isnull().sum()
```
#判断每行每列，若为null则返回True，返回narray
>>>iris.isnull()
[
    [False,False,False,False,False],
    [False,False,False,False,False],
    ....
]

>>>iris.isnull().sum()
sepal length (cm)    0
sepal width (cm)     0
petal length (cm)    0
petal width (cm)     0
target               0
dtype: int64

#查看数据缺失的行
>>>iris[iris.isnull().values==True]
```
-------------------
> 检查分类是否均匀 pd.DataFrame.groupby([...]).count() [[详细文档]](http://pandas.pydata.org/pandas-docs/version/0.17.0/generated/pandas.DataFrame.groupby.html)
```
#统计结果类别数量是否比较均衡
>>>iris.groupby('y').count()
target sepal length (cm)  sepal width (cm)  petal length (cm)  petal width (cm)
0.0                 50                50                 50                50
1.0                 50                50                 50                50
2.0                 50                50                 50                50

>>> iris.groupby(['y','sepal length (cm)']).count()
target   sepal length (cm)   sepal width (cm)  petal length (cm)  petal width (cm)
0.0 4.3                               1                  1                 1
    4.4                               3                  3                 3
    4.5                               1                  1                 1
...
1.0 4.9                               1                  1                 1
    5.0                               2                  2                 2
    5.1                               1                  1                 1
...
2.0 4.9                               1                  1                 1
    5.6                               1                  1                 1
    5.7                               1                  1                 1
...
```

> 条件挑选 pd.DataFrame[条件][[列名]]
```
#挑出y等于0，其中的sepal length和y列
>>>iris[iris['target']==0][['sepal length (cm)','target']]
    sepal length (cm)    target
0                 5.1  0.0
1                 4.9  0.0
2                 4.7  0.0
...
```
-----------------------
> 随机挑选 pd.DataFrame.sample(n=2 or frac=0.1) [[详细文档]](http://pandas.pydata.org/pandas-docs/version/0.17.0/generated/pandas.DataFrame.sample.html)
```
#随机返回两行数据
>>> iris.sample(2) # or iris.sample(n=2)
#随机返回10%行数据
>>> iris.sample(frac=0.1) #要求必须加frac=，且n和frac不可同时使用
```
---------------
> 划分训练集与测试集 train_test_split() [[详细文档]](http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html#sklearn.model_selection.train_test_split)
```
from sklearn.cross_vaildation import train_test_split
x_train, y_train, x_test, y_test = train_test_split(
    iris[sk_iris.feature_names], 
    iris['target'],
    test_size= 0.3
)
#参数说明
> *array 可以输入一个或多个数组进行划分
> test_size or train_size 
    可以是int、float、None。int抽取指定数目；float抽取指定比例；
    其中一个为None，则该值与非None值互补；
    .若两个皆为None，则默认test_size为0.25,train_size为0.75；
> random_state 随机种子。相同值返回的结果相同，不同值或未指定，返回的结果不同。默认None
> shuffle 若为True，则先打乱排列再划分；否则按顺序划分。默认True
```