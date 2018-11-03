# author    TuringEmmy
# time      2018/10/19 20:38
# project   MachineLearning

from sklearn.decomposition import PCA

"""
主成分分析
"""


def pca():
    """
    主成分分析进行特征降维
    :return: None
    """
    pca = PCA(n_components=0.9)

    data_source = [
        [2, 8, 4, 5],
        [6, 3, 0, 8],
        [5, 4, 9, 1]
    ]
    data = pca.fit_transform(data_source)
    print(data)

    return None


if __name__ == '__main__':
    pca()
