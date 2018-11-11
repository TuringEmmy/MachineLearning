# author    TuringEmmy
# time      2018/10/19 20:56
# project   MachineLearning

import pandas as pd
from sklearn.decomposition import PCA

# 读取四张表的数据
prior = pd.read_csv('./data/instacart/order_products_prior.csv')

products = pd.read_csv('./data/instacart/producte.csv')

orders = pd.read_csv('./data/instacart/orders.csv')

aisies = pd.read_csv("./data/instacart/aisies.csv")

# 进行合并四张表到一张表（用户-物品类别）
_mg = pd.merge(prior, products, on=['pro'])

pd.merge(_mg, orders, on=['order_id', 'order_id'])

pd.merge(_mg, aisies, on=['asile_id', 'order_id'])

mt = pd.merge(_mg,aisies,on=['asile_id','aisle_id'])


print(mt.head(10))

# 交叉表(特殊的分组工具)

cross = pd.crosstab(mt['user_id','asile'])

print(cross.head(10))

# 进行主成分分析(降维)
pca = PCA(n_components=0.9)

data = pca.fit_transform(cross)
print(data)
print(data.shape)








