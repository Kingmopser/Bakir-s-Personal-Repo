import pandas as pd
from src.preprocess import CleanerImport
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import root_mean_squared_error
from sklearn.feature_selection import f_regression
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_classif

# getting dataset
filePathTest  = "/Users/kingmopser/PycharmProjects/Housing_Prices_Prediction/assets/test.csv"
filePathTrain = "/Users/kingmopser/PycharmProjects/Housing_Prices_Prediction/assets/train.csv"

#apply cleaining
train = CleanerImport(filePathTrain)
test = CleanerImport(filePathTest)
X_train, X_test, y_train, y_test = train_test_split(X= train,y= train["SalePrice"], train_size= 0.8, test_size=0.2, random_state=12)

# Feature Selection



bestFeatures= SelectKBest(score_func=f_regression, k=30)
res = bestFeatures.fit_transform(X=train,y=train["SalePrice"])

# remove features

train.columns[bestFeatures.get_support()]


