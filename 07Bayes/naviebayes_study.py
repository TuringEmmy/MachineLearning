# author    TuringEmmy
# time      2018/10/26 17:08
# project   MachineLearning
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

from sklearn.naive_bayes import MultinomialNB


def naviebayes():
    """朴素贝叶斯进行文本分类"""
    news = fetch_20newsgroups(subset='all')

    # 进行数据分割

    x_train, x_test, y_train, y_test = train_test_split(news.data, news.target, test_size=0.25)

    # 对数据集进行特征抽取
    tf = TfidfVectorizer()

    # 以训练集当中的词的列表进行每篇文章重要性性统计 ['a','b','c']
    x_train = tf.transform(x_train)
    print(tf.get_feature_names())

    x_test = tf.transform(x_test)

    # 激进型朴素贝叶斯算的预测alpha是拉普拉斯系数
    mlt = MultinomialNB(alpha=1.0)

    print("x的训练集：", x_train.toarray())
    # 利用历史数据进行培训
    mlt.fit(x_train, y_train)

    # 进行预测
    y_predict = mlt.predict(x_test)

    # 得出准确率
    print("准确率：", mlt.score(x_test, y_predict))
    return None


if __name__ == '__main__':
    naviebayes()
