from sklearn.svm import SVR
import numpy as np
import pandas as pd
import sys
DATA_UTIL_PATH = "../"
sys.path.append(DATA_UTIL_PATH)
import datautil
from joblib import dump, load
VAL_RATIO = 0.25
import time
from sklearn.metrics import mean_squared_error, make_scorer
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import expon, reciprocal

full_data = pd.read_csv('../train.csv')
full_data = full_data[(full_data['adr'] < 1000) & (full_data['adr'] > -100)] # remove outliers
full_data = full_data[full_data['is_canceled'] == 0] # only use the uncanceled orders to train

# get the preprocessor and the default training features
preprocessor, features_spec = datautil.get_the_data_preprocessor()

# split data into input and label
X_train_full_raw = full_data[features_spec]
y_train_full = np.array(full_data['adr'])
X_transformed = preprocessor.fit_transform(X_train_full_raw)

param_dist = {'kernel':['linear','rbf'],
                  'C':reciprocal(1,100),
                  'gamma':expon(scale=1.0)}
                  
regr = RandomizedSearchCV(SVR(), param_distributions=param_dist, n_iter=100, cv=3,
                               scoring='neg_mean_squared_error', verbose=3, n_jobs=-1, random_state=42)
regr.fit(X_transformed,y_train_full)
print('Best parameters: ' + str(regr.best_params_))
print('Best score: ' + str(regr.best_score_))

# save grid search 
dump(regr.best_estimator_, 'svm.model')

