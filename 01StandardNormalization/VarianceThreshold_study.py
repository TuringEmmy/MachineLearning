# author    TuringEmmy
# time      2018/10/18 23:08
# project   MachineLearning

from sklearn.feature_selection import VarianceThreshold


# -------------------------------方差特征选择-------------------------
def var():
    """
    特征选择-删除弟方差的特征
    :return:
    """
    vari = VarianceThreshold(threshold=1.0)
    reso = [
        [0, 2, 0, 3],
        [0, 1, 4, 3],
        [0, 1, 1, 3]
    ]
    data = vari.fit_transform(reso)

    print(data)


if __name__ == '__main__':
    var()
