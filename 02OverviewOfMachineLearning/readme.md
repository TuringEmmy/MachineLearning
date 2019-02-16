
在sklearn中，估计器(estimator)是一个重要的而角色，是一类实现了算法的API

1. 用于分类的估计器
- sklearn.neighbors k紧邻算法
- sklearn.naive_bayes 贝叶斯
- sklearn.linear_model.LogisticRefression 逻辑回归
- sklearn.tree  决策树与随机森林

2.  用于回归的估计器
- sklearn.linear.modle.linearRegression  线性回归
- sklearn.linear_model.Ridge  岭回归

---
每个算法的API当中的参数，必须搞清楚每个算法原理

### 估计器

训练集  测试集


1. 调用fit
fit(x_train,y_train)

**estimator**  

2. 输入与测试集的数据


### 机器学习的基础

1. 机器学习开发流程
算法是核心，数据和计算是基础
找准定位
大部分复杂模型的算法设计都是算法工程师在做，而我们
- 分析很多的数据
- 分析具体的业务
- 应用常见的算法
- 特征工程、调参数、优化

我们应该怎么做

学会分析问题，使用机器学习算法的目的，想要算法完成何种任务

- 掌握算法基本思想，学会对问题用相应的算法解决

- 学会利用库或者框架解决问题


原始数据明确问题做什么
数据的基本处理：pd去处理（缺失值，合并表。。）
特征工程（特征积极性处理）
- 分类
- 回归
找到合适的算法积进行预测
模型的评估，判定效果


`上线使用，以API形式提供`

2. 模型是什么

``
3. 机器学习算法分类









查询了网络，没有找到相关的解决办法，只好自己琢磨解决。

解决办法如下：
1）下载20news-bydate.tar.gz
2）下载20news-bydate.pkz
以上两个文件直接在网络上搜索，有很多链接的。

3）在~\scikit_learn_data\20news_home 下解压20news-bydate.tar.gz，有2个目录： 20news-bydate-test和20news-bydate-train
~表示什么就不用多说了吧。

4）拷贝20news-bydate.pkz到~\scikit_learn_data\下面，并改名为：20news-bydate_py3.pkz
这就是最关键的一步，一定要改名。








