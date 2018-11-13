# author    TuringEmmy
# time      2018/11/13 17:29
# project   MachineLearning

"""
逻辑分类跟线性回归是一样的，
Z(w) = w_0 = w_1 * x_1 + ...

sigmoid 分类 ，进行分类成两类

逻辑回归进行癌症的预测
实例数据699个，来进行纵六判断，它是恶性的还是良性的

也就是常说你的二分分类问题，判定概率值，那个类别少，判定概率值的这个类别

"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# 分类评估里
from sklearn.metrics import classification_report


def logistic_handle():
    """
    
    逻辑回归做二分类进行癌症预测（根据细胞的属性特征）
    :return: 
    """
    # 构造列标签名字(opandas 默认第一行为字段名称，如果数据里面没有，则要进行指定)
    column = ['Sample code number', 'Clump Thickness', 'Uniformity of Cell Size', 'Uniformity of Cell Shape',
              'Marginal Adhesion', 'Single Epithelial Cell Size', 'Bare Nuclei', 'Bland Chromatin', 'Normal Nucleoli',
              'Mitoses', 'Class']

    # 读取数据
    data = pd.read_csv(
        "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data",
        names=column)

    print(data.head())

    # +++++++++++++++++++++++缺失值进行处理（这里的缺失值是?）
    data = data.replace(to_replace="?", value=np.nan)

    # 也可以把以前的数据删除
    data = data.dropna()

    # ++++++++++++++++++++数据进行分割++++++++++++++++++++++
    x_train, x_test, y_train, y_test = train_test_split(data[column[1:10]], data[column[10]],
                                                        test_size=0.25)  # 10不包括,后面这个是取第11列的数据

    # 进行标准化处理
    std = StandardScaler()

    x_train = std.fit_transform(x_train)
    x_test = std.transform(x_test)

    # 逻辑回归预测
    lg = LogisticRegression(C=1.0)
    lg.fit(x_train, y_train)

    print(lg.coef_)

    y_predict = lg.predict(x_test)

    print('准确率：', lg.score(x_test, y_test))

    print("召回率:", classification_report(y_test, y_predict, labels=[2, 4], target_names=["良性", "恶性"]))

    return None


if __name__ == '__main__':
    logistic_handle()
