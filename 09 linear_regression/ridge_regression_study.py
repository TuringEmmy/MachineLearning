# author    TuringEmmy
# time      2018/11/11 19:12
# project   MachineLearning



from sklearn.datasets import load_boston
# 波斯顿房价
from sklearn.linear_model import SGDRegressor, LinearRegression,Ridge
from sklearn.metrics import mean_squared_error

from sklearn.model_selection import train_test_split

# 标准化预处理
from sklearn.preprocessing import StandardScaler


# 预测房子价格
def myLinear():
    """
    线性回归直接预测房子价格
    :return:
    """

    # 获取数据
    lb = load_boston()
    # 分割数据集到训练集合测试集
    x_train, x_test, y_train, y_test = train_test_split(lb.data, lb.target, test_size=0.25)

    print(y_train, y_test)
    # 进行标准化处理  (？)目标值是否需要标准化处理？为什么需要？
    # 特征值和目标值都是必须进行标准化处理，要实例化两个保准API
    std_x = StandardScaler()
    x_train = std_x.fit_transform(x_train)
    x_test = std_x.transform(x_test)

    # 目标值
    std_y = StandardScaler()

    # *******************0.19版本必须传入二维的数据************************
    # y_train = std_y.fit_transform(y_train.reshape(-1, 1))
    y_train = std_y.fit_transform(y_train.reshape(1, -1))

    y_test = std_y.transform(y_test)

    # estimator预测
    # 正规方程求解方式预测结果
    lr = LinearRegression()

    lr.fit(x_train, y_train)

    print(lr.coef_)

    # 预测测试集的房子就价格
    y_lr_predict = std_y.inverse_transform(lr.predict(x_test))

    print('正规方程测试集里面每个房子的预测价格', y_lr_predict)

    print('正规方程的均方误差:', mean_squared_error(std_y.inverse_transform(y_test), y_lr_predict))
    # 领回归去进行房价的预测
    rd = Ridge(alpha=1.0)
    # alpha力度

    rd.fit(x_train, y_train)
    print(rd.coef_)

    # 预测测试集的房子的价格
    y_rd_predict = std_y.inverse_transform(rd.predict(x_test))

    print('梯度下降测试集里面每个房子的预测价格', y_rd_predict)

    print('梯队下降的均方误差:', mean_squared_error(std_y.inverse_transform(y_test), y_rd_predict))
    return None


if __name__ == '__main__':
    myLinear()
