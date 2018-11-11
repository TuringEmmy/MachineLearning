# author    TuringEmmy
# time      2018/10/30 22:52
# project   MachineLearning

import pandas as pd

# 导入字典抽取
from sklearn.feature_extraction import DictVectorizer
from sklearn.cross_validation import train_test_split

# 随机森林
from sklearn.ensemble import RandomForestClassifier

# 决策树
from sklearn.grid_search import GridSearchCV
from sklearn.tree import DecisionTreeClassifier, export_graphviz


def decision():
    # 决策树泰坦尼克号的预测生死
    titanic = pd.read_csv("http://biostat.mc.vanderbilt.edu/wiki/pub/Main/DataSets/titanic.txt")

    # 处理数据，找出特征值和目标值
    x = titanic[['pclass', 'age', 'sex']]
    y = titanic['survived']

    print(x)
    # 缺失值处理，inplae 是把控制钱聪
    x['age'].fillna(x['age'].mean(), inplace=True)

    print(y)
    # 分割数据集到训练集合测试集
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)

    # 进行处理(特征工程)，特征-》类型——》one_hot编码
    # 字典特征抽取
    dict = DictVectorizer(sparse=False)

    # 这个是默认的写法，把一行转化成字典
    x_train = dict.fit_transform(x_train.to_dict(orient='records'))

    print(dict.get_feature_names())

    x_test = dict.transform(x_test.to_dict(orient='records'))

    # 随机森林进行
    rf = RandomForestClassifier()

    param = {
        'n_estimators': [120, 200, 300, 500, 800, 1200],
        "max_depth": [5, 8, 15, 25, 30]
    }

    # 网格搜索与交叉验证
    gc = GridSearchCV(rf, param_grid=param, cv=2)

    gc.fit(x_train, y_train)

    # 预测准确率
    print("准确率:", gc.score(x_test, y_test))

    # 查看选择的参数的模型
    print("查看选择的参数的模型：", gc.best_params_)
    return None


if __name__ == '__main__':
    decision()
