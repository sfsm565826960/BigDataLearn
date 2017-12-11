# 数据处理I：缺失值填补

> 下列数据来源[Kaggle的Titanic题目](https://www.kaggle.com/prkukunoor/TitanicDataset/data)

## 查找缺失值

> 统计数据缺失 pd.DataFrame.isnull().sum()
```
# 判断每行每列，若为null则返回True，返回narray
>>>data = pd.read_csv(DATA_FILE)
>>>data.isnull().sum()
PassengerId      0
Survived         0
Pclass           0
Name             0
Sex              0
Age            177
SibSp            0
Parch            0
Ticket           0
Fare             0
Cabin          687
Embarked         2
dtype: int64
```
> 查看数据缺失行
```
# 查看全部有缺失的行
>>>data[data.isnull().values == True]

# 查看年龄有缺失的行
>>>data[data.Age.isnull().values == True]
```
-------------------

## 缺失值填补
要点：尽量保持原始信息状态。<br>
方法：
- 当缺少比例很小时，可以直接扔掉这部分样本数据；
- 按某个统计量补全，可以是定值、均值、中位数等；
- 拿函数或模型预测缺失值。

> 删除缺失样本
```
# 只要该行存在缺失值，就将该行删除
data.dropna(axis='index', how='any', inplace=True)

# 删除Embarked缺失的整行数据
data = data[data.Embarked.notnull()]
```
[pandas.DataFrame.dropna](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.dropna.html)

> 定值填充
```
# 用定值填充
data.Age = data.Age.fillna(0) # 缺失值被填充为0
# 同 data.Age.fillna(0, inplace=True)
# inplace为true表示在原数组修改数据，不返回拷贝数组

# 用前后有效值填充
# ffill 前一个有效值，bfill后一个有效值
data.Age = data.Age.fillna(method='ffill')
```
[pandas.DataFrame.fillna](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.fillna.html)<br>

> 统计量填充
```
# 均值填充
data.loc[(data.Age.isnull()), 'Age'] = data.Age.dropna().mean()

# 使用Imputer填充
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)
data = imputer.fit_transform(data)
```
[pandas.DataFrame.loc](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.loc.html)<br>
[sklearn.preprocessing.Imputer](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.Imputer.html#sklearn.preprocessing.Imputer)<br>
[sklearn.preprocessing.Imputer中文](http://blog.csdn.net/kancy110/article/details/75041923)

> 预测值填充（略）
* 可以用函数进行填补
* 可以针对该特征建立模型进行预测