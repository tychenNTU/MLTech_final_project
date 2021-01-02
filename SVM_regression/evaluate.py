import sys
DATA_UTIL_PATH = "../"
sys.path.append(DATA_UTIL_PATH)
from joblib import dump, load
from sklearn.metrics import mean_squared_error
import datautil
import pandas as pd
import numpy as np
import time

VAL_RATIO = 0.25

full_data = pd.read_csv('../train.csv')
full_data = full_data[(full_data['adr'] < 1000) & (full_data['adr'] > -100)] # remove outliers
full_data = full_data[full_data['is_canceled'] == 0] # only use the uncanceled orders to train
# get the preprocessor and the default training features
preprocessor, features_spec = datautil.get_the_data_preprocessor()

# split data into input and label
X_train_full_raw = full_data[features_spec]
y_train_full = np.array(full_data['adr'])
X_transformed = preprocessor.fit_transform(X_train_full_raw)

# load model
regr = load("svm.model")

# make prediction on the test data
start = time.time()
y_pred = regr.predict(X_transformed)       
time_spent = time.time() - start
print("Total time spent for prediction: (seconds) " + str(time_spent))

# print the MSE
MSE = mean_squared_error(y_train_full, y_pred)
print("MSE: " + str(MSE))

# compute the predicted label
# revenue_df = datautil.get_revenue_df(test_data)
# revenue_df["revenue_label"] = revenue_df["revenue"].apply(lambda revenue: int(revenue/10000))

# save model
# dump(regr, "svm.model")