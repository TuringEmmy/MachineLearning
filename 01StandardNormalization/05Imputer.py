# author    TuringEmmy
# time      2018/10/18 22:14
# project   MachineLearning


from sklearn.preprocessing import Imputer
import numpy as np


# ---------------------------------------------缺失值处理----------------------------
def im():
    # strategy填补策略
    # missing_values田步志
    # axis=0是列
    im = Imputer(missing_values='NaN', strategy='mean', axis=0)
    data = im.fit_transform([[1, 2], [np.nan, 3], [7, 6]])
    print(data)
    return None


if __name__ == '__main__':
    im()

# 打印结果是：说明1+7=8. 8/2 = 4
# [[1. 2.]
#  [4. 3.]
#  [7. 6.]]