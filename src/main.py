
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import root_mean_squared_error

# reading Dataset
test = pd.read_csv("/Users/kingmopser/PycharmProjects/Housing_Prices_Prediction/assets/test.csv")
train = pd.read_csv("/Users/kingmopser/PycharmProjects/Housing_Prices_Prediction/assets/train.csv")


# overview of dataset
print(train.head(5))
print(train.tail(5))
print(train.info)
print(train.dtypes)
print(train.isna().sum())

# checking for duplicates
train[train.duplicated()].shape

#checking which columns have NAs
na_col = train.isnull().sum().loc[lambda x: x>0 ]
na_col

# replacing NA placeholer with Strings, since they arent invalid entries.
# it just stands for "No.." validating with Text book

train["Alley"] = train["Alley"].replace(0,"No")


