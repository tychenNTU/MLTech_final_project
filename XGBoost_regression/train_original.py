import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import xgboost
from sklearn.model_selection import GridSearchCV
import pickle as pkl
import os, sys
from sklearn.model_selection import cross_val_score
sys.path.append(os.path.realpath('..'))

with open("../preprocess/processed_train.pkl", "rb") as f:
    global train_x, train_y
    train_x, train_y = pkl.load(f)
                    
cv_results = cross_val_score(xgboost.XGBRegressor(), train_x, train_y, 
                            cv=3, scoring="neg_mean_squared_error", verbose=3, n_jobs=-1)
mean_score = round(np.mean(cv_results), 4)
print("MSE of the default XGBoost model: " + str(mean_score*-1))