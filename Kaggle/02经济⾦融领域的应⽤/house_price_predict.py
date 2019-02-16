# life is short, you need use python to create something!
# author    TuringEmmy
# time      12/31/18 3:57 PM
# project   Kaggle

import numpy as np
import pandas as pd

train_df = pd.read_csv('./house_price/input/train.csv',index_col=0)
test_df = pd.read_csv('./house_price/input/test.csv', index_col=0)

# print(train_df.head())
prices = pd.DataFrame({"price":train_df["SalePrice"], "log(price + 1)": np.log1p(train_df['SalePrice'])})
prices.hist()
y_train = np.log1p(train_df.pop('SalePrice'))
all_df = pd.concat((train_df, test_df), axis=0)
print(all_df.shape)

from sklearn.ensemble import RandomForestRegressor

RandomForestRegressor(n_estimators=2,max_features=)