
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
from sklearn.feature_selection import mutual_info_classif
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
plt.show()
msno.matrix(train)
plt.show()
msno.heatmap(train)
plt.show()

# preprocessing
def Cleaner(df):
    const_imputer = SimpleImputer(strategy='constant', fill_value="unknown")
    mean_imputer = SimpleImputer(strategy='mean')
    encoder = LabelEncoder()
    col = df.select_dtypes(include=["object"]).columns
    # Imputing NAs
    for i in df.columns :
        if df.dtypes[i] == 'object':
           df[[i]] = const_imputer.fit_transform(df[[i]])
        else:
            df[[i]] =mean_imputer.fit_transform(df[[i]])

    # defining ordered columns via domain knowledge
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

    #applying order to columns
    for k,v in ord_mappings.items():
        df[k]= pd.Categorical(df[k], categories=v, ordered=True)

    # applying encoding
    df[col] =df[col].apply(encoder.fit_transform)
    return df


# applying both functions
train = Cleaner(train)
test = Cleaner(test)

# Feature Selection

X=train.drop(columns="SalePrice")
y=train["SalePrice"]
f_value = f_regression(X, y)
MI_score = mutual_info_classif(X, y, random_state=0)
for i,j in enumerate(X.columns):
    print(f"{j} {MI_score[i]}")
