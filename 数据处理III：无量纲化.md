# 数据处理III：无量纲化


--------------
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