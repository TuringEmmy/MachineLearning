# author    TuringEmmy
# time      2018/10/18 21:05
# project   MachineLearning


# 特征预处理
# 通过特定的统计方法（数学方法）将数据转换成算法要求的数据
# 数值型数据：标准缩放
# 1、 归一化
# 2、标准化

# 类型型数据：one-hot编码

# 时间类型：时间地切片

# sklearn.preprocessing
# 归一化处理
from sklearn.preprocessing import MinMaxScaler


# ------------------------------------归一化处理---------------------------
def normalization():
    """
    归一化处理
    :return: None
    """
    # mm = MinMaxScaler()
    # 设置默认区间设置
    mm = MinMaxScaler(feature_range=(2, 3))

    data = mm.fit_transform([[90, 2, 10, 40], [60, 4, 15, 45], [75, 3, 13, 46]])

    print(data)
    return None


if __name__ == '__main__':
    normalization()
