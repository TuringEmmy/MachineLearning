# author    TuringEmmy
# time      2018/11/11 14:38
# project   MachineLearning


"""
预测房价的增长变化规律
"""
from matplotlib import pyplot as plt

plt.figure(figsize=(10, 10))
x = [60, 72, 75, 80, 83]
y = [126, 152, 151, 157.5, 168]
plt.scatter(x, y)
# plt.savefig('./房价散点图.png')
plt.show()
