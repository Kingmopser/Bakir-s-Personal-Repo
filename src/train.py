from src.preprocess import preprocessing
import pandas as pd
import numpy as np
from sklearn.metrics import root_mean_squared_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV

# getting dataset
filePathTest  = "/Users/kingmopser/PycharmProjects/Housing_Prices_Prediction/assets/test.csv"
filePathTrain = "/Users/kingmopser/PycharmProjects/Housing_Prices_Prediction/assets/train.csv"
if __name__ == "__main__":
    #apply cleaining
    train = preprocessing(filePathTrain,"train")
    test = preprocessing(filePathTest,"test")


def training(df):

    # seed
    np.random.seed(23)

    # train, test split
    X_train, X_test, y_train, y_test = train_test_split(df.drop(columns="SalePrice",inplace = False),df["SalePrice"],
                                                        train_size= 0.8, test_size=0.2)
    # initializing model: randomforest
    rf = RandomForestRegressor(n_estimators=100)
    rf.fit(X_train,y_train)
    predictions = rf.predict(X_test)
    print(f"RMSE:{ root_mean_squared_error(y_test,predictions)} for initial model")

    # Hypertuning

    # dictionary for possible parameter values
    d1 = {"max_depth" : np.arange(1,20), "max_features" : ["sqrt", "log2", None]}
    # rndm Grid Search 10 CV for hyperparameters
    rdSearch = RandomizedSearchCV(rf, param_distributions=d1,cv=10,n_jobs=-1,random_state=23)
    rdSearch.fit(X_train,y_train)
    predictionsrd =rdSearch.predict(X_test)
    print(f"RMSE:{ root_mean_squared_error(y_test,predictionsrd)} for tuned model, fitted on full train set")

    model = rdSearch.best_estimator_
    model.fit(df.drop(columns = "SalePrice",inplace=False), df["SalePrice"])

    return model


















