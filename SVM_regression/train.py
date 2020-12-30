from sklearn.svm import SVR
import numpy as np
import pandas as pd
import sys
DATA_UTIL_PATH = "../"
sys.path.append(DATA_UTIL_PATH)
import datautil
from joblib import dump, load
VAL_RATIO = 0.25
import util
import time

full_data = pd.read_csv('../train.csv')
full_data = full_data[(full_data['adr'] < 1000) & (full_data['adr'] > -100)] # remove outliers
full_data = full_data[full_data['is_canceled'] == 0] # only use the uncanceled orders to train

# get the preprocessor and the default training features
preprocessor, features_spec = datautil.get_the_data_preprocessor()

# split data into input and label
X_train_full_raw = full_data[features_spec]
y_train_full = np.array(full_data['adr'])
X_transformed = preprocessor.fit_transform(X_train_full_raw)

# split the data into training set and validation set
n_data = len(full_data)
shuffled_indices = np.random.permutation(n_data)
val_set_size = int(n_data * VAL_RATIO)
val_indices = shuffled_indices[:val_set_size]
train_indices = shuffled_indices[val_set_size:]
X_train, y_train = X_transformed[train_indices], y_train_full[train_indices]
X_val, y_val = X_transformed[val_indices], y_train_full[val_indices]

regr = SVR(C=1.0, epsilon=0.2)

# fit model
start = time.time()
regr.fit(X_train, y_train)
stop = time.time()
print(f"Training time: {stop - start}s")

# score
print("R^2 score: " + str(regr.score(X_val, y_val)))
print("MAE: " + str(util.MAE(regr.predict(X_val), y_val)))

# calculate label 
# rf_clf = load("../RF_cls_and_NN_reg/rf_cls.model")

# # make prediction on the test data
# test_data["is_canceled"] = rf_clf.predict(X_test_rf) # predicted is_canceld
# test_data["adr"] = regr.predict(X_val)         # predicted adr

# # compute the predicted label
# revenue_df = datautil.get_revenue_df(test_data)
# revenue_df["revenue_label"] = revenue_df["revenue"].apply(lambda revenue: int(revenue/10000))

# save model
dump(regr, "svm.model")

