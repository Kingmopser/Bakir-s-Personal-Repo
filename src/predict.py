from src.train import training
from src.preprocess import preprocessing
import pandas as pd
import numpy as np
# getting dataset
filePathTrain = "/Users/kingmopser/PycharmProjects/Housing_Prices_Prediction/assets/train.csv"
filePathTest  = "/Users/kingmopser/PycharmProjects/Housing_Prices_Prediction/assets/test.csv"

preIndex = pd.read_csv(filePathTest)["Id"]
train = preprocessing(filePathTrain,"train")
test = preprocessing(filePathTest,"test")
test = test[train.columns[:-1]]

# Best estimator applied on test
model = training(train)
predictionsBest =model.predict(test)
predDf= pd.DataFrame(preIndex,columns=["Id"])
predDf["SalePrice"] = predictionsBest

predDf.to_csv("/Users/kingmopser/PycharmProjects/Housing_Prices_Prediction/assets/submission.csv", index=False)