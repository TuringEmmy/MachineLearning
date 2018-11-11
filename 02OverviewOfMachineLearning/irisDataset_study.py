# author    TuringEmmy
# time      2018/10/21 21:51
# project   MachineLearning

from sklearn.datasets import load_iris

# 导入划分数据集的
from sklearn.model_selection import train_test_split

li = load_iris()

# # 特征值
# print(li.data)
# # 目标值
# print(li.target)
# # 描述
# print("---DESCR---")
# print(li.DESCR)
# 特征名,新闻数据，手写数字、回归数据集没有
# print("---feature_names---")
# print(li.feature_names)
# print('----target_names----')
# print(li.target_names)

# 返回值   训练集train x_train y_train 和测试集test x_test y_test
x_train, x_test, y_train, y_test = train_test_split(li.data, li.target, test_size=0.27)

print("训练集特征值和目标值:", x_train, y_train)
print('*' * 50)
# 意义就是那一部分训练集乱序的充当测试集
print("测试集特征值和目标值:", x_test, y_test)
