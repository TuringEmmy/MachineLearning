# author    TuringEmmy
# time      2018/10/29 15:45
# project   MachineLearning

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV,train_test_split

from sklearn.neighbors import KNeighborsClassifier


def naviebayes():
    """朴素贝叶斯进行文本分类"""
    from sklearn.datasets import fetch_20newsgroups
    news = fetch_20newsgroups(subset='all')

    # 进行数据分割

    x_train, x_test, y_train, y_test = train_test_split(news.data, news.target, test_size=0.25)

    # 对数据集进行特征抽取
    tf = TfidfVectorizer()

    # 以训练集当中的词的列表进行每篇文章重要性性统计 ['a','b','c']
    x_train = tf.transform(x_train)
    print(tf.get_feature_names())

    x_test = tf.transform(x_test)

    knn = KNeighborsClassifier()

    # 构造一些超参数的值进行搜索
    param = {
        "n_neighbors": [3, 5, 10]
    }
    # 进行网格搜索

    gc = GridSearchCV(knn, param_grid=param, cv=2)
    """
    """

    # 预测准确率
    print("在测试集上的准确率：", gc.score(x_test, y_test))

    print("在交叉验证当中的最好结果:", gc.best_score_)

    print('选择最好的模型结果:', gc.best_estimator_)

    print('每个超参数每次交叉验证的结果:', gc.cv_results_)

    return None


if __name__ == '__main__':
    naviebayes()
