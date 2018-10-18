# author    TuringEmmy
# time      2018/10/18 20:49
# project   MachineLearning、

# TF: term frequency：词的频率
# IDF：inverse document frequency: 文档频率  log(总文档数量、该词出现的文档数量)

# log(数组)：随着输入的数值越小，结果越小

# tf* idf  重要性程度
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer


def cutword():
    con1 = jieba.cut("今天很残酷，明天更残酷，后天很美好，但绝对大部分是死在明天晚上，所以每个人不要放弃今天。")

    con2 = jieba.cut("我们看到的从很远星系来的光是在几百万年之前发出的，这样当我们看到宇宙时，我们是在看它的过去。")

    con3 = jieba.cut("如果只用一种方式了解某样事物，你就不会真正了解它。了解事物真正含义的秘密取决于如何将其与我们所了解的事物相联系。")

    # 转换成列表
    content1 = list(con1)
    content2 = list(con2)
    content3 = list(con3)

    # 把列表转换成字符串
    c1 = ' '.join(content1)
    c2 = ' '.join(content2)
    c3 = ' '.join(content3)

    return c1, c2, c3


def tfidfvec():
    c1, c2, c3 = cutword()
    print(c1, c2, c3)
    cv = TfidfVectorizer()
    data = cv.fit_transform([c1, c2, c3])
    print(cv.get_feature_names())
    print(data.toarray())
    return None


if __name__ == '__main__':
    tfidfvec()

# 注意;tf-idf相乘的结果，面试会问到
