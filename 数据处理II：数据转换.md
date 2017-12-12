# 数据处理II：数据转换
> 下列数据来源[Kaggle的Titanic题目](https://www.kaggle.com/prkukunoor/TitanicDataset/data)

> 特征分类

- 定量特征：如年龄、票价等有数量关系的特征，可**二值化**或**函数变换**
- 定性特征：如性别、几等舱等没有数量意义的特征，可**哑编码**或**函数变换**

&emsp;&emsp;定量特征 与 定性特征 需要分开处理

-------------

> 二值化 [Binarizer](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.Binarizer)

定量特征二值化的核心在于设定一个阈值，大于阈值的赋值为1，小于等于阈值的赋值为0
```
from sklearn.preprocessing import Binarizer
binarizer = Binarizer(threshold=18, copy=True)
data['Adult'] = binarizer.fit_transform(data['Age'])

# 等同
data['Adult'] = (data['Age'] > 18).astype(int)
```

----------

> 哑编码 [OneHotEncoder](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder)

&emsp;&emsp;数据集中有很多变量类型是Nominal的，对于某些算法而言，完全的Nominal类型的变量是无法计算的，这就需要进行哑编码。

&emsp;&emsp;哑编码是一种状态编码，可将定性特征转为数字，但这些数字没有大小关系。该特征有N类就有N位，某位对应一类，是该类标则该位标1，其余位标0，如：`001`，`010`，`100`，最终形成稀疏矩阵。

```
from sklearn.preprocessing import OneHotEncoder

# 几等舱Pclass(1等舱, 2等舱, 3等舱)
data['Pclass'] = OneHotEncoder().fit_transform(data['Pclass'].reshape(-1 ,1))

# 输出
print data['Pclass'].toarray()

[ [ 0.  0.  1.]
       ...
  [ 0.  1.  0.] ]
```

&emsp;&emsp;这里遇到一个问题：OneHotEncoder编码Embarked会报错`ValueError: could not convert string to float: Q`，暂时未找到解决方法，若要好的解决方法请留言，谢谢。

> 重建定性特征 [pandas.get_dummies](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.get_dummies.html)

- 定性特征只有`是`与`非`，因此也可拆分为多个二值化特征：
```
dummies_pclass = pd.get_dummies(data['Pclass'], prefix='Pclass')

print dummies_pclass.head(2)
|Pclass_1   |Pclass_2   |Pclass_3   |
|   0.0     |   0.0     |   1.0     |
|   1.0     |   0.0     |   0.0     |
```
- 若像性别一样，只有两种类别，且非此即彼，则可：
```
data.loc[data.Sex == 'female', 'Sex'] = 1
data.loc[data.Sex == 'male', 'Sex'] = 0

print data.Sex.head(2)
|  Sex  |
|   0   |
|   1   |
```
- 对于缺失的特征，重建特征后每个标签值都为0：
```
# Embarked 登船港口有两条数据为null，重建特征后每个标签值为0
dummies_embarked = pd.get_dummies(data['Embarked'], prefix='Embarked')

print dummies_embarked.loc[61] #其中Embarked为nan的数据
|Embarked_C |Embarked_Q |Embarked_S |
|   0.0     |   0.0     |   0.0     |
```

---------------

> 构造非线性特征

原因：线性模型就是把特征对分类结果的作用加起来（例如：Y = T1 + T2），但线性模型无法表示一些非线性的关系（如：Y = T1 * T2），所以我们打算人工构造一些新特征，弥补线性模型对非线性表达式表达能力的不足。

特征的非线性的表达式可以分两种：
- 表达“数值特征”本身的非线性因素。
```
#方式一：将原有数值的高次方作为特征，x^y或lnx
df['Age*Age_scaled'] = scaler.fit_transform(data['Age'] * data['Age'])

#方式二：数据离散化（将连续的数值划分为区间）
df['Child'] = (data['Age'] <= 18).astype(int)
```
- 表达特征与特征之间存在的非线性关联，并且这种关联关系对分类结果有帮助。
```
#设想头等舱的人能受到更好的保护待遇，同时年龄越小的越优先被救助，所以这两个特征是否有一些关联
df['Age*Pclass_scaled'] = scaler.fit_transform(data['Age'] * data['Pclass'])
```

----------------------

> 特征合并、删除、筛选
```
df = pd.concat([df, dummies_embarked, dummies_sex, dummies_pclass, dummies_cabin], axis=1) #axis=0(index),1(columns)

df.drop(['Pclass','Name','Sex', 'Ticket','Cabin','Embarked','Fare', 'Age'], axis=1, inplace=True)

df = df.filter(regex='Survived|Age_.*|SibSp|Parch|Child|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')

np_array = df.as_matrix() #将DataFrame转为Numpy-array
```
[pandas.concat](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.concat.html)<br>
[pandas.DataFrame.drop](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.drop.html)<br>
[pandas.DataFrame.filter](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.filter.html)<br>
[pandas.DataFrame.as_matrix](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.as_matrix.html)