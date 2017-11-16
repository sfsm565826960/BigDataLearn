### numpy用法
numpy是快速操作结构数组的工具，经常和pandas结合在一起用（[查看网页](http://www.cnblogs.com/prpl/p/5537417.html)）

> np.r_[] 和np.c_[]
```
>>> a = np.array([1,2,3])
>>> b = np.array([5,2,5])

>>> //测试 np.r_ 按行组合
>>> np.r_[a,b]
array([1, 2, 3, 5, 2, 5])

>>> //测试 np.c_ 按列组合
>>> np.c_[a,b]
array([[1, 5],
       [2, 2],
       [3, 5]])
       
>>> np.c_[a,[0,0,0],b]
array([[1, 0, 5],
       [2, 0, 2],
       [3, 0, 5]])
```

> np.append
```
>>> np.append([1, 2, 3], [[4, 5, 6], [7, 8, 9]])
array([1, 2, 3, 4, 5, 6, 7, 8, 9])

>>> np.append([[1, 2, 3], [4, 5, 6]], [[7, 8, 9]], axis=0)
array([[1, 2, 3],
       [4, 5, 6],
       [7, 8, 9]])
       
>>> np.append([[1, 2, 3], [4, 5, 6]], [7, 8, 9], axis=0)
Traceback (most recent call last):
...
ValueError: arrays must have same number of dimensions
```

> np.dot
```
import numpy as np

a = np.array([[1,2,3], [4,5,6]])
b = np.array([[1,2], [3,4], [5,6]])

c = np.dot(a, b) # 矩阵乘法
[[22 28]
 [49 64]]

d = c + 1 
[[23 29]
 [50 65]]
```

> np.array.reshape
```
# 重新调整矩阵的行数、列数
a = np.array([
        [1,2,3],
        [4,5,6]
    ])

print a.reshape(3, 2)
print a.reshape(3, -1) #列自动计算
print a.reshape(-1, 2) #行自动计算
[[1 2]
 [3 4]
 [5 6]]
```

> np.mean 和 np.std 归一化
```
a = np.array([0.1,0.2,0.3])
b = np.array([10,20,30])

print np.mean(a) #均值 0.2
print np.std(a) #方差 0.0816496580928

## 归一化：让数据满足均值为0，方差相似。可以使梯度下降更快。

a_scaled = (a - np.mean(a)) / np.std(a)
b_scaled = (b - np.mean(b)) / np.std(b)

## 或者

import sklearn.preprocessing as preprocessing
scaler = preprocessing.StandardScaler()
a_scaled2 = scaler.fit_transform(a)
b_scaled2 = scaler.fit_transform(b)

```

> np.sum
```
>>> np.sum([[0, 1], [0, 5]], axis=None)
6
>>> np.sum([[0, 1], [0, 5]], axis=0)
array([0, 6])
>>> np.sum([[0, 1], [0, 5]], axis=1)
array([1, 5])
```