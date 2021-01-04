import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import xgboost
from sklearn.model_selection import RandomizedSearchCV
import pickle as pkl
import os, sys
from joblib import dump, load
sys.path.append(os.path.realpath('..'))

with open("../preprocess/processed_train.pkl", "rb") as f:
    global train_x, train_y
    train_x, train_y = pkl.load(f)

param_grid = {
    'max_depth': [6, 10, 15, 20],
    'learning_rate': [0.001, 0.01, 0.1, 0.2, 0,3],
    'subsample': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    'colsample_bytree': [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    'colsample_bylevel': [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    'min_child_weight': [0.5, 1.0, 3.0, 5.0, 7.0, 10.0],
    'gamma': [0, 0.25, 0.5, 1.0],
    'reg_lambda': [0.1, 1.0, 5.0, 10.0, 50.0, 100.0],
    'n_estimators': [100]
}

gsearch1 = RandomizedSearchCV(estimator = xgboost.XGBRegressor(), param_distributions = param_grid, verbose=3,scoring='neg_mean_squared_error', cv=3, n_iter=100, random_state=42, n_jobs=-1)
gsearch1.fit(train_x,train_y)
print('best params')
print (gsearch1.best_params_)
print('best score')
print (gsearch1.best_score_)

# save grid search 
dump(gsearch1.best_estimator_, 'xgboost.model')