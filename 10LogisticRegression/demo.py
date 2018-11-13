# author    TuringEmmy
# time      2018/11/13 20:04
# project   MachineLearning
from sklearn.externals import joblib
from sklearn.metrics import classification_report

y_true = [0, 1, 2, 3, 2]
y_pred = [0, 0, 2, 2, 1]
target_names = ['class 0', 'class 1', 'class 2', 'class 3']
# 其中的target_names代表真实值和与测试里面共有多少个类
print(classification_report(y_true, y_pred, target_names=target_names))


y_true = [0, 1, 2, 2, 2]
y_pred = [0, 0, 2, 2, 1]
target_names = ['class 0', 'class 1', 'class 2']
print(classification_report(y_true, y_pred, target_names=target_names))

joblib.load()