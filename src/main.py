
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import root_mean_squared_error
from sklearn.feature_selection import f_regression


# reading Dataset
test = pd.read_csv("/Users/kingmopser/PycharmProjects/Housing_Prices_Prediction/assets/test.csv")
train = pd.read_csv("/Users/kingmopser/PycharmProjects/Housing_Prices_Prediction/assets/train.csv")


# overview of dataset
print(train.head(5))
print(train.tail(5))
print(test.info)
print(train.dtypes)
print(train.isna().sum())

# checking for duplicates
train[train.duplicated()].shape

# handling missing data/imputing


# Function replacing NA placeholer with Strings for categorical variables and with 0 for numerical
# since they aren't invalid entries.
# it just stands for "No" validating with code book

def rep(dataframe):
    for i in dataframe.columns :
        if not dataframe.dtypes[i] == "int64":
           dataframe[i] = dataframe[i].fillna(value="No")
        else:
            dataframe[i] = dataframe[i].fillna(value=0)

# for train set
rep(train)
# for train set
rep(test)

# Feature Selection with Random Forest

f_val = f_regression()
print(test[test["Utilities"].isna()])


























