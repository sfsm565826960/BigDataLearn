### numpy用法
numpy是快速操作结构数组的工具，经常和pandas结合在一起用（[查看网页](http://www.cnblogs.com/prpl/p/5537417.html)）

> ndarray

&emsp;&emsp;多维数组对象，可以直接参与运算。它是一个通用的**同构数据**多维容器，即其中所有元素必须是相同类型的，每个数组都有一个`shape`（表示各纬度大小的元组）和一个`dtype`（表明数组元素数据类型的对象）。

> ndarray的创建
```
array = [1,2,3]
narray = np.array(array) # array([1, 2, 3])
np.array(array, dtype= np.float64) #创建时指定数据类型
# np.asarray 区别：如果输入本身是ndarray就不进行复制，直接返回本身；而array都复制返回

zero1 = np.zeros(2) # array([0, 0])
zero2 = np.zeros((2, 3)) # 创建2行3列全为0的数组
zero3 = np.zeros_like(narray) #创建形状与narray类似的全为0的数组
# np.ones() np.ones_like() 可创建全为1的数组
# np.empty() np.empty_like() 可以创建空的数组，但不是nan、None或0，而是未初始化的垃圾值

np.arange(3) # array([0, 1, 2])，创建时也可以指定dtype=float来创建float类型的数组

np.eye(N) # 和np.identity(N) 一样创建N*N的单位矩阵（对角线为1，其余为0）
```

> 数据类型
- int8、uint8 (同8,16,32,64)有符号、无符号的n位整数
- float16 (同16,32,64,128) n位浮点数
- complex64 (同64,128,256) n位复数
- bool 布尔值
- object 对象类型
- string_ 字符串类型
- unicode_ unicode类型

> 显式转换dtype
```
narray2 = narray1.astype(np.float64) 
# 也可简单写成narray.astype(float)
# astype返回复制而不是本身，即使dtype与原来相同也如此，
# 因此narray1的类型不变，narray2为转换类型后的narray1的复制
```

> narray可以直接运算
```
na = np.arange(3, dtype=float) # [0., 1., 2.]
na + 1       # array([1., 2., 3.])
na * na      # array([0., 1., 4.]) 等同 na ** 2
1 / (na + 1) # array([1., 0.5, 0.3333333333333])
# 也可以直接进行逻辑运算
na > 1       # array([False, False, True])
```

> narray多维数组
```
# 创建
arr3d = np.array([[[1, 2], [3, 4]],[[5, 6], [7, 8]]])

# 索引
arr3d[0][1][1]
arr3d[0,1,1] # 简化，python的list不支持

# 布尔索引，要求布尔narray形状和narray一致。python的list不支持。
bools = [[[True, False], [False, True]], [[True, True], [False, False]]]
nbools = np.array(bools) # 不支持python的布尔数组索引，需转为narray
arr3d[nbools]            # [1, 4, 5, 6]
arr3d[arr3d > 4]         # [5, 6, 7, 8]，由narray的逻辑运算的布尔索引
arr3d[(arr3d > 2) & (arr3d < 6)] # [3 4 5]，布尔索引支持&和、|或等运算
arr3d[(arr3d < 2) | (arr3d > 6)] # [1 7 8]，注意and和or在布尔索引运算里无效

# 花式索引：为了以特定的顺序选取行子集，需传入用于指定顺序的整数列表或narray
arr3d[[1, 0]]            # array([[[5, 6], [7, 8]], [[1, 2], [3, 4]]])
arr3d[1, [-1, 0]]        # array([[7, 8], [5, 6]])，支持负数索引
# 传入多个整数列表有些特别：
arr3d[[1, 0], [1, 0]]    # array([[7, 8], [1, 2]])，即返回[arr3d[1, 1], arr3d[0, 0]]
# 若想对行和列以特定顺序选取，有两种方式：
arr3d[[1, 0]][:, [1, 0]] # array([[[7, 8], [5, 6]], [[3, 4], [1, 2]]])
arr3d[np.ix_([1, 0], [1, 0])] # 同上

# 赋值，标量和数组都可以
arr3d[0, 1] = 3      # array([[[1, 2], [3, 3]],[[5, 6], [7, 8]]])
arr3d[0, 1] = [4, 4] # array([[[1, 2], [4, 4]],[[5, 6], [7, 8]]])
# 布尔索引不支持数组赋值
arr3d[arr3d < 5] = 0 # array([[[0, 0], [0, 0]],[[5, 6], [7, 8]]])
```

> narray切片索引 [[python的list切片索引见网页]](https://www.liaoxuefeng.com/wiki/001374738125095c955c1e6d8bb493182103fac9270762a000/0013868196352269f28f1f00aee485ea27e3c4e47f12bc7000)

narray的切片与list的切片类似，但也有不同之处：
- **narray的切片的返回并非复制，任何修改都会直接反映到源数组上！**
    
    原因：NumPy设计目的是处理大数据，若每次切片都复制会造成性能和内存问题。

    ```
    src = np.arange(5)
    # src = array([0, 1, 2, 3, 4])

    src[2:4] = 5
    # src = array[0, 1, 5, 5, 4])，而python的list不支持这种操作。

    part = src[0:2]
    part[1] = 5
    # part = array[0, 5]), src = array[0, 5, 5, 5, 4])

    # 若想要得到复制，则需显示复制：
    part = src[0:2].copy(
    ```

- narray还支持多轴切片：
    ```
    list = [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]
    nlist = np.array(list)
    nlist[1:, :1]   # array([[[5 6]]])，python的list不支持多轴切片
    nlist[:, :1]    # array([[[1, 2]], [[5, 6]]])，单:表示选取整个轴
    nlist[1, :1]    # array([[5 6]])，数字索引和切片混合可以降低纬度
    nlist[np.array([False, True]), :1]  # array([[[5 6]]])，布尔索引不能降低纬度
    ```

> narray转置和轴对换
```
# 对于二维矩阵，有个特殊T属性
arr = np.arange(9).reshape(3, 3)
arr.T

# 对于高维矩阵，可以用transpose或swapaxes
arr = np.arange(16).reshape(2,2,4)
a1 = arr.transpose((1, 0, 2)) # 转置
a2 = arr.swapaxes((1, 2))     # 轴对换
```

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