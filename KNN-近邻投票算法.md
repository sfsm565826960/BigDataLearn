## KNN-近邻算法

> ### 原理
&emsp;&emsp;KNN模型是选出与待分类样本最相似的K个近邻，投票确定样本所属分类。

&emsp;&emsp;例子：如下图，绿色圆要被决定赋予哪个类，是红色三角形还是蓝色四方形？

<center><img src="http://images0.cnblogs.com/blog2015/771535/201508/041623504236939.jpg" height="150" width="150" /></center>

&emsp;&emsp;如果K=3，由于红色三角形所占比例为2/3，绿色圆将被赋予红色三角形那个类；如果K=5，由于蓝色四方形比例为3/5，因此绿色圆被赋予蓝色四方形类。

在KNN中，具体执行步骤：
1.  计算待测样本和训练集中每个样本点的欧式距离
2.  对上面所有距离值排序
3.  选前K个最小距离的样本作为选民

> ### 特点
1. 典型非参数模型，计算机通过KNN学到的知识并不是以权重的方式存储下来的；
2. 原理简单；
3. 保存模型需要保存全部样本集；
4. 训练过程很快，预测速度很慢。（主要原因是必须对数据集中的每一个数据计算距离值）

> ### 运用
1. **思考并发扬模型的特征。** 近邻分类的思路使KNN在近邻聚集比较好的数据任务中表现良好。如判断文本是新闻还是娱乐；若任务类别不清晰，如预估用户是否点击一个事件或预测股票市场则表现较差。

2. **规避工程执行的短处。** KNN的工程问题在于预测时间复杂度高，处理大数据时要注意。如稀疏特征值可以考虑做索引表；密集特征值可以考虑kd-tree等数据结构。

3. **优化时寻找模型的关键位置。** KNN模型优化突破口在于相似度的计算，即如何合理地划分出近邻。

-----------------------

## 投票KNeighborsClassifier

> ### 前提
1. 所选择的邻居都是已经正确分类的对象；
2. 基本上邻居们是均等权重投票。（特定场景也可以设置权重规则）

> ### 数据预处理
```
# 0. 加载数据
import numpy as np
import pandas as pd
from sklearn import datasets
scikit_iris = datasets.load_iris();
iris = pd.DataFrame(
    data=np.c_[scikit_iris['data'], scikit_iris['target']],
    columns=np.append(scikit_iris.feature_names, ['target'])
)

# 1. 观察数据
iris.head(2)

# 2. 检查数据是否有缺失
iris.isnull().sum()

# 3. 观察样本是否均匀
iris.groupby('target').count()

# 4. 划分训练集和测试集
from sklearn.cross_validation import train_test_split
train_data, test_data, train_target, test_target = train_test_split(
    scikit_iris['data'],
    scikit_iris['target'],
    test_size=0.3,
    random_state=0
)
```
```
>【sum与count的区别】

  # sum   求和小计
  >>>iris.isnull().sum()
  sepal length (cm)    0
  sepal width (cm)     0
  petal length (cm)    0
  petal width (cm)     0
  target               0
  dtype: int64
  
  # count 统计非NaN的数量
  >>>iris.isnull().count()
  sepal length (cm)    150
  sepal width (cm)     150
  petal length (cm)    150
  petal width (cm)     150
  target               150
  dtype: int64
  
  #注：isnull()返回的True和False被识别为0和1
```

> ### 训练模型
KNeighborsClassifier [[详细文档]](http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier)
```
常用参数说明：
n_neighbors {int}       选择多少邻居进行投票，默认5。
weights     {str|fun}   权重
            'uniform' 相同的(默认);
            'distance' 按距离减小; 编写自定义函数
```
```
#KNN近邻投票算法
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1) #只选择一个邻居
knn.fit(train_data, train_target) #训练模型

#预测新数据
test_predict = knn.predict(test_data)
```

> ### 评估模型
```
from sklearn import metrics
print metrics.accuracy_score(test_target, test_predict)
```