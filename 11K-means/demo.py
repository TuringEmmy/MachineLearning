# author    TuringEmmy
# time      2018/11/13 21:27
# project   MachineLearning


import numpy as np

from sklearn.cluster import KMeans

X = np.array([[1, 2], [1, 4], [1, 0],[4, 2], [4, 4], [4, 0]])
kmeans = KMeans(n_clusters=2, random_state=0)

kmeans.fit(X)
result = kmeans.predict([[0, 0], [4, 4]])
print(result)
print(kmeans.cluster_centers_)
print(kmeans.labels_)
#
# X = np.array([
#     [1, 2],
#     [1, 4],
#     [1, 0],
#     [4, 2],
#     [4, 4],
#     [4, 0]])
# kmeans = KMeans(n_clusters=2, random_state=0)
# # kmeans.cluster_centers_
# kmeans.labels_
# print(kmeans.predict([[0, 0], [4, 4]]))