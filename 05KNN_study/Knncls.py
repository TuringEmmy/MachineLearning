# author    TuringEmmy
# time      2018/10/26 14:56
# project   MachineLearning
'''
对签到的预测
'''

import pandas as pd
from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsClassifier



def knncls():
    """
    k-紧邻算法预测用户签到的位置
    :return: None
    """
    # 读取数据
    data = pd.read_csv("./data/FBlocation/train.csv")

    # print(data.head())

    # 处理数据
    # 1. 缩小数据,查询数据筛选
    data = data.query("x > 1.0 & x < 1.25 & y > 2.5 & y < 2.75")

    # 处理时间的数据
    time_value = pd.to_datetime(data['time'], unit='s')
    # print(time_value)

    # 构造一些特征
    data['day'] = time_value.day
    data['hour'] = time_value.hour
    data['weekday'] = time_value.weekday

    # 把时间戳特征删除
    # 这里列是1哦，注意pd的每次操作都是有返回值的哦
    data.drop(['time'], axis=1)

    print(data)

    # 把签到数量少于n个目标的位置删除
    place_count = data.groupby('palce_id').count()

    tf = place_count[place_count.row_id > 3].reset_index()

    data = data[data['place_id'].isin(tf.place_id)]

    # 取出数据特征++22的当中的特征值
    y = data['place_id']

    x = data.drop(['place_id'], axis=1)

    # 进行数据的分割训练集和测试集
    x_train, x_test, y_test = train_test_split(x, y, test_size=0.25)

    # 特诊工程(标准化)

    # # 进行算法流程
    # ==========================================================
    knn = KNeighborsClassifier(n_neighbors=5)

    # 得出预测结果
    y_predict = knn.predict(x_test)

    print('预测的目标标签的位置为：', y_predict)

    # 得出准确率
    print('预测的准确率:', knn.score(x_test, y_test))
    # ===========================================================

    return None


if __name__ == '__main__':
    knncls()
