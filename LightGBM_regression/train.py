import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import lightgbm as lgb
from sklearn.model_selection import RandomizedSearchCV
import pickle as pkl
from scipy.stats import randint as sp_randint
from scipy.stats import uniform as sp_uniform
import os, sys
from joblib import dump, load
sys.path.append(os.path.realpath('..'))

with open("../preprocess/processed_train.pkl", "rb") as f:
    global train_x, train_y
    train_x, train_y = pkl.load(f)

param_grid = {
    'max_depth': sp_randint(10,50),
    'num_leaves': sp_randint(6, 50), 
    'min_child_samples': sp_randint(100, 500), 
    'min_child_weight': [1e-5, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4],
    'subsample': sp_uniform(loc=0.2, scale=0.8), 
    'colsample_bytree': sp_uniform(loc=0.4, scale=0.6),
    'reg_alpha': [0, 1e-1, 1, 2, 5, 7, 10, 50, 100],
    'reg_lambda': [0, 1e-1, 1, 5, 10, 20, 50, 100]
}

gsearch1 = RandomizedSearchCV(estimator = lgb.LGBMRegressor(), param_distributions = param_grid, n_jobs=-1, verbose=3,scoring='neg_mean_squared_error', cv=3, n_iter=100, random_state=42)
gsearch1.fit(train_x,train_y)
print('best params')
print (gsearch1.best_params_)
print('best score')
print (gsearch1.best_score_)

# save grid search 
dump(gsearch1.best_estimator_, 'lightgbm.model')