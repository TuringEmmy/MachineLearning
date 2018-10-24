# author    TuringEmmy
# time      2018/10/21 22:06
# project   MachineLearning
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split

# all表示下载所有的数据
news = fetch_20newsgroups(subset="all")
print(news.data)
print(news.target)
