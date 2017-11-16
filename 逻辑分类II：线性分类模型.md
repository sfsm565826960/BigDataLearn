# 逻辑分类II：线性分类模型

> ### 灵感：偏导数
对于寻找使函数最小的参数问题，导数具有“引路”作用。

[例子] 寻找y=x^2中使y最小的x
```
import numpy as np

#y = 1 * x^2 + 0 * x + 0 = x^2
y = np.poly1d([1, 0, 0])

#y的导函数
d_yx = np.polyder(y)

# 当d_yx小于0，函数趋小的区域；
# 当d_yx大于0，函数趋大的区域；
# 在函数最小值处，d_yx等于0
```
1. 随机选一个起点 (x0, y0)
```
import random

x0 = random.uniform(-10, 10)
y0 = random.uniform(-10, 10)
```
2. 计算当前点导数，按导数“指引”，往前走一步，到达一个新点(x1, y1)
```
def step(x, d_yx):
    alpha = 0.2 #学习速率
    return x - alpha * d_yx(x)
step(x0, d_yx)
```
3. 不断重复第二步
```
x = x0
x_list = [x]
for i in range(10):
    x = step(x, d_yx)
    x_list.append(x)
print x_list
```

&emsp;&emsp;可以看到`x_list`中`x`的值越来越接近0(最优解)，最终在0附近来回震荡，这个现象叫作收敛。

&emsp;&emsp;由于底部震荡，因此最终得到的`x`不一定等于最佳参数，但我们可以忽略这种损失，比起暴力尝试所有参数组合，能够在有限的执行步骤中找到一个“相对合适”的参数已经到达我们的目标了。

> ### 寻找模型参数组合

&emsp;&emsp;回到【逻辑分类I：线性分类模型】中寻找`min(loss(w,b))`的问题，虽然参数从1个变成2个，但我们依旧可以按之前的思路设计流程。
