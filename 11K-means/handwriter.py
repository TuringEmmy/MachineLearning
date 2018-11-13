# author    TuringEmmy
# time      2018/11/13 21:38
# project   MachineLearning
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from sklearn import datasets

def kmeans():
    """
    手写数字聚类过程
    :return: None
    """
    # 加载数据

    ld = datasets.load_digits()

    print(ld.target[:20])

    # 聚类
    km = KMeans(n_clusters=810)

    km.fit_transform(ld.data)

    print(km.labels_[:20])

    print(silhouette_score(ld.data, km.labels_))

    return None


if __name__ == "__main__":
    kmeans()
