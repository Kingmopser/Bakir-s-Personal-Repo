
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import root_mean_squared_error
from sklearn.feature_selection import f_regression
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
import missingno as msno

# reading Dataset
test = pd.read_csv("/Users/kingmopser/PycharmProjects/Housing_Prices_Prediction/assets/test.csv")
train = pd.read_csv("/Users/kingmopser/PycharmProjects/Housing_Prices_Prediction/assets/train.csv")

# overview of dataset
print(train.head(5))
print(train.tail(5))
print(test.info())
print(train.dtypes)
print(train.isna().sum())

# visualizing missing data
train['SalePrice'].value_counts()
train.isnull().mean().loc[lambda x: x>0 ].sort_values(ascending=False)
msno.bar(train)
msno.matrix(train)
msno.heatmap(train)
plt.show()

# imputing NAs

# Function imputing dataframe using SimpleImputer
def rep(dataframe):
    const_imputer = SimpleImputer(strategy='constant', fill_value="unknown")
    mean_imputer = SimpleImputer(strategy='mean')
    for i in dataframe.columns :
        if dataframe.dtypes[i] == 'object':
           dataframe[[i]] = const_imputer.fit_transform(dataframe[[i]])
        else:
            dataframe[[i]] =mean_imputer.fit_transform(dataframe[[i]])
    return dataframe

#Function ordering and apply Label Encoder
def OrderingAndEncoding(df):
    ord_mappings = {
        "LotShape": ["IR3","IR2","IR1","Reg"],
        "LandSlope": ["Sev","Mod","Gtl"],
        "ExterQual": ["Po", "Fa", "TA", "Gd", "Ex"],
        "ExterCond": ["Po", "Fa", "TA", "Gd", "Ex"],
        "BsmtQual": ["unknown", "Po", "Fa", "TA", "Gd", "Ex"],
        "BsmtCond": ["unknown", "Po", "Fa", "TA", "Gd", "Ex"],
        "BsmtExposure": ["unknown", "No", "Mn", "Av", "Gd"],
        "BsmtFinType1": ["unknown", "Unf", "LwQ", "Rec", "BLQ", "ALQ", "GLQ"],
        "BsmtFinType2": ["unknown", "Unf", "LwQ", "Rec", "BLQ", "ALQ", "GLQ"],
        "HeatingQC": ["Po", "Fa", "TA", "Gd", "Ex"],
        "KitchenQual": ["Po", "Fa", "TA", "Gd", "Ex"],
        "Functional": ["Sal", "Sev", "Maj2", "Maj1", "Mod", "Min2", "Min1", "Typ"],
        "FireplaceQu": ["unknown", "Po", "Fa", "TA", "Gd", "Ex"],
        "GarageFinish": ["unknown", "Unf", "RFn", "Fin"],
        "GarageQual": ["unknown", "Po", "Fa", "TA", "Gd", "Ex"],
        "GarageCond": ["unknown", "Po", "Fa", "TA", "Gd", "Ex"],
        "PoolQC": ["unknown", "Po", "Fa", "TA", "Gd", "Ex"],
        "Fence": ["unknown", "MnWw", "GdWo", "MnPrv", "GdPrv"]
    }
    encoder = LabelEncoder()
    col = list(ord_mappings.keys())
    df[col] =df[col].apply(encoder.fit_transform)

# One-Hot-Encoder for Categorical
    df = pd.get_dummies(df,drop_first=True)
    return df


# applying both functions
train = OrderingAndEncoding(rep(train))
test = OrderingAndEncoding(rep(test))

# Feature Selection


