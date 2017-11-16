# 数据预处理II：特征处理
> 下列数据来源[Kaggle的Titanic题目](https://www.kaggle.com/prkukunoor/TitanicDataset/data)

> 特征分类

- 数值特征：如年龄、票价等有数量关系的特征
- 类别特征：如性别、几等舱等没有数量意义的特征
- 数值特征 与 类别特征 需要分开处理

--------------
> 处理数值特征缺失

要点：尽量保持原始信息状态。<br>
方法：
- 当缺少比例很小时，可以直接扔掉这部分样本数据；
- 按某个统计量补全，可以是定值、均值、中位数等；
- 拿模型预测缺失值。
```
#查看数据缺失的行
print data[data.isnull().values == True] #查看全部有缺失的行
print data[data.Age.isnull().values == True] #查看年龄有缺失的行

#删除缺失样本
data = data[data.Age.notnull()]

#定值填充
data.Age = data.Age.fillna(0)
#或 data.Age.fillna(0, inplace=True)

#均值填充
data.loc[(data.Age.isnull()), 'Age'] = data.Age.dropna().mean()

#用前后值填充
#ffill 前一个有效值，bfill后一个有效值
data.Age = data.Age.fillna(method='ffill')

#取出某些值
data = data[data.Age.isin(range(10,20))) #取出10-19岁的用户
data = data[data.Age > 20] #取出20岁以上的用户
```
[pandas.DataFrame.fillna](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.fillna.html)<br>
[pandas.DataFrame.loc](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.loc.html)<br>
[pandas.DataFrame.dropna](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.dropna.html)<br>
[pandas.DataFrame.isin](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.isin.html)<br>
[range](http://www.runoob.com/python/python-func-range.html)
> 数值特征归一化

若几种数值特征不在同一可以比较的尺度上，则需要进行归一化
```
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
df['Age_scaled'] = scaler.fit_transform(data['Age']) #年龄
df['Fare_scaled'] = scaler.fit_transform(data['Fare']) #票价
```
由于版本不同，有时会报错：`Expected 2D array, got 1D array instead`，我们可以暂用`data.Age.values.reshape(-1,1)`进行处理。
```
df['Age_scaled'] = scaler.fit_transform(data['Age'].values.reshape(-1,1)) #年龄
df['Fare_scaled'] = scaler.fit_transform(data['Fare'].values.reshape(-1,1)) #票价
```
[sklearn.preprocessing.StandardScaler](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html)<br>
[numpy.reshape](https://docs.scipy.org/doc/numpy/reference/generated/numpy.reshape.html)<br>
[Expected 2D array, got 1D array instead](http://blog.csdn.net/llx1026/article/details/77940880)

------------------------
> 处理类别特征缺失
```
# 客舱号Cabin缺少程度太严重，数据不可能有效恢复。
# 虽然Cabin无法提取，但可以转换为“有没有在客舱”
# 提醒：注意代码顺序不能反
data.loc[data.Cabin.notnull(), 'Cabin'] = 'Yes' #在客舱
data.loc[data.Cabin.isnull(), 'Cabin'] = 'No' #不在客舱
```
> 重建类别特征

类别特征只有`是`与`不是`两个值。<br>
因此对于几等舱Pclass(1等舱, 2等舱, 3等舱)等类别特征，需要按类重建特征：
```
dummies_pclass = pd.get_dummies(data['Pclass'], prefix='Pclass')

print dummies_pclass.head(2)
|Pclass_1   |Pclass_2   |Pclass_3   |
|   0.0     |   0.0     |   1.0     |
|   1.0     |   0.0     |   0.0     |
```
[pandas.get_dummies](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.get_dummies.html)

若像性别一样，只有两种类别，且非此即彼，则可：
```
data.loc[data.Sex == 'female', 'Sex'] = 1
data.loc[data.Sex == 'male', 'Sex'] = 0

print data.Sex.head(2)
|  Sex  |
|   0   |
|   1   |
```
然而经测试，重建列表特征有利于提高准确率，因此尽量进行重建：
```
dummies_sex = pd.get_dummies(data['Sex'], prefix='Age')
dummies_cabin = pd.get_dummies(data['Cabin'], prefix='Cabin')
```
对于缺失的特征，重建特征后每个标签值都为0：
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