k-means通常被称为劳埃德算法，这在数据聚类中是最经典的，也是相对容易理解的模型。算法执行的过程分为4个阶段。

- 1.首先，随机设K个特征空间内的点作为初始的聚类中心。
- 2.然后，对于根据每个数据的特征向量，从K个聚类中心中寻找距离最近的一个，并且把该数据标记为这个聚类中心。
- 3.接着，在所有的数据都被标记过聚类中心之后，根据这些数据新分配的类簇，通过取分配给每个先前质心的所有样本的平均值来创建新的质心, 重新对K个聚类中心做计算。
- 4.最后，计算旧和新质心之间的差异,如果所有的数据点从属的聚类中心与上一次的分配的类簇没有变化，那么迭代就可以停止，否则回到步骤2继续循环。

K均值等于具有小的全对称协方差矩阵的期望最大化算法

**sklearn.cluster.KMeans**

```python
class sklearn.cluster.KMeans(n_clusters=8, init='k-means++', n_init=10, max_iter=300, tol=0.0001, precompute_distances='auto', verbose=0, random_state=None, copy_x=True, n_jobs=1, algorithm='auto')
  """
  :param n_clusters:要形成的聚类数以及生成的质心数

  :param init:初始化方法，默认为'k-means ++',以智能方式选择k-均值聚类的初始聚类中心，以加速收敛;random,从初始质心数据中随机选择k个观察值（行

  :param n_init：int，默认值：10使用不同质心种子运行k-means算法的时间。最终结果将是n_init连续运行在惯性方面的最佳输出。

  :param n_jobs：int用于计算的作业数量。这可以通过并行计算每个运行的n_init。如果-1使用所有CPU。如果给出1，则不使用任何并行计算代码，这对调试很有用。对于-1以下的n_jobs，使用（n_cpus + 1 + n_jobs）。因此，对于n_jobs = -2，所有CPU都使用一个。

  :param random_state:随机数种子，默认为全局numpy随机数生成器
  """
```

**example**

```python
X = np.array([[1, 2], [1, 4], [1, 0],[4, 2], [4, 4], [4, 0]])
kmeans = KMeans(n_clusters=2, random_state=0)
```

**方法**

```python
kmeans.fit(X)
```

| method  | use                              | detail        |
| ------- | -------------------------------- | ------------- |
| fit     | fit(X,y=None)                    | 使用X作为训练数据拟合模型 |
| predict | kmeans.predict([[0, 0], [4, 4]]) | 预测新的数据所在的类别   |

```python
kmeans.predict([[0, 0], [4, 4]])
# array([0, 1], dtype=int32)
```

**属性**

| 属性               | use                     | detail   |
| ---------------- | ----------------------- | -------- |
| cluster_centers_ | kmeans.cluster_centers_ | 集群中心的点坐标 |
| labels_          | kmeans.labels_          | 每个点的类别   |

```
kmeans.cluster_centers_
array([[ 1.,  2.],
       [ 4.,  2.]])
kmeans.labels_
```