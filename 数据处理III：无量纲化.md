# 数据处理III：无量纲化

> 下列data数据来源[Kaggle的Titanic题目](https://www.kaggle.com/prkukunoor/TitanicDataset/data)

- 特征的规格不一样，不能够放在一起比较。
- 无量纲化可将不同规格的数据转化为同一规格的数据。

```
# 测试数据：
na = np.asarray([
  [1., 10., -100.],
  [2., 20., -200.],
  [3., 30., -300.]
])
```

--------------

> 标准化：[StandardScaler](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html)

**前提**：特征值服从正态分布。
```
from sklearn.preprocessing import StandardScaler
print StandardScaler().fit_transform(na)

[[-1.22474487 -1.22474487  1.22474487]
 [ 0.          0.          0.        ]
 [ 1.22474487  1.22474487 -1.22474487]]
```

```
data[['Age', 'Fare']] = StandardScaler().fit_transform(data[['Age', 'Fare']])

# 或者
data['Age_scaled'] = scaler.fit_transform(data['Age']) # 年龄
data['Fare_scaled'] = scaler.fit_transform(data['Fare']) # 票价
```
由于版本不同，有时传入1维数组时会报错：`Expected 2D array, got 1D array instead`，我们可以暂用`data.Age.values.reshape(-1,1)`进行处理。
```
data['Age_scaled'] = scaler.fit_transform(data['Age'].values.reshape(-1,1))
data['Fare_scaled'] = scaler.fit_transform(data['Fare'].values.reshape(-1,1))
```
[numpy.reshape](https://docs.scipy.org/doc/numpy/reference/generated/numpy.reshape.html)<br>
[Expected 2D array, got 1D array instead](http://blog.csdn.net/llx1026/article/details/77940880)

-----------

> 区间放缩
- [MinMaxScaler](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html) 将数据缩放至[0,1]的区间
```
# MinMaxScaler = (x - X.min) / (X.max - X.min)

from sklearn.preprocessing import MinMaxScaler
print MinMaxScaler().fit_transform(na)

[[ 0.   0.   1. ]
 [ 0.5  0.5  0.5]
 [ 1.   1.   0. ]]
```
- [MaxAbsScaler](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MaxAbsScaler.html) 将最大的绝对值缩放至单位大小
```
# MaxAbsScaler = x / np.absolute(X).max

from sklearn.preprocessing import MaxAbsScaler
print MaxAbsScaler().fit_transform(na)

[[ 0.33333333  0.33333333 -0.33333333]
 [ 0.66666667  0.66666667 -0.66666667]
 [ 1.          1.         -1.        ]]
```

----------------

> 归一化：[Normalizer](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.Normalizer.html)

归一化与上两者的**区别**：
* 上两者按特征矩阵的`列`处理数据，归一化按特征矩阵的`行`处理数据；
* 上两者让`列与列`进行向量运算时具有同一标准，归一化让`行与行`进行向量运算时具有统一标准。

```
from sklearn.preprocessing import Normalizer
print Normalizer().fit_transform(na)

[[ 0.00994988  0.09949879 -0.99498793]
 [ 0.00994988  0.09949879 -0.99498793]
 [ 0.00994988  0.09949879 -0.99498793]]
```