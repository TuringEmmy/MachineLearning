# author    TuringEmmy
# time      2018/10/17 22:42
# project   MachineLearning

from sklearn.feature_extraction import DictVectorizer


# 特征抽取
#
# 导入包
# from sklearn.feature_extraction.text import CountVectorizer
#
# # 实例化CountVectorizer
#
# vector = CountVectorizer()
#
# # 调用fit_transform输入并转换数据
#
# res = vector.fit_transform(["life is short,i like python","life is too long,i dislike python"])
#
# # 打印结果
# print(vector.get_feature_names())
#
# print(res.toarray())

# 字典数据抽取
def dictvec():
    """
    字典数据抽取
    :return:None
    """
    # 实例化sparse简写矩阵，节约内存
    dict = DictVectorizer(sparse=False)
    # 调用fit_transform
    data = dict.fit_transform(
        [{'city': '北京', 'temperature': 100}, {'city': '上海', 'temperature': 60}, {'city': '深圳', 'temperature': 30}])

    # 把字典中一些类别的数据分别进行转换成特征，是数值类型的就不再转换了
    print(dict.get_feature_names())
    print(data)

    # 打印每个特征值是多少
    print(dict.inverse_transform(data))
    return None


from sklearn.feature_extraction.text import CountVectorizer

# 对文本进行特征化
def countvec():
    """
    对文本进行特征化
    :return: None
    """
    cv = CountVectorizer()
    data = cv.fit_transform(["I like china ,i live in hanzhong", "I work in beijing, i come from xian"])

    # 统计所有文章当中所有的词，重复的只看一次，词的列表
    # 对每篇文章，在词的列表里面进行统计每个词传的次数
    # 单个字母不统计
    print(cv.get_feature_names())
    # print(data)
    print(data.toarray())

    # 总结：
    # 文本特征抽取：Count              [文本分类，情感分析]
    # 对于单个英文字母不同： 没有分类的依据
    # 对于中文，按逗号进行的，不支持哦，如果词语之间实用空格，可实现效果，单个汉字也是不统计的哦！！呵呵呵，机器学习真好玩
    return None

import jieba
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

# 中文特征是化
def hanzivec():
    """
    中文特征值化
    :return: None
    """
    c1, c2, c3 = cutword()
    print(c1, c2, c3)
    cv = CountVectorizer()
    data = cv.fit_transform([c1, c2, c3])
    print(cv.get_feature_names())
    print(data.toarray())
    return None


if __name__ == '__main__':
    # dictvec()
    # countvec()
    hanzivec()