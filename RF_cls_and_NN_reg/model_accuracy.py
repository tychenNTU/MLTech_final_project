# common:
import pandas as pd
import numpy as np
import sys
# for ML:
import tensorflow.keras as keras
DATA_UTIL_PATH = "../"
sys.path.append(DATA_UTIL_PATH)
import datautil
import joblib
import tensorflow.keras as keras 

VAL_RATIO = 0.25 # tatio for validation data

full_data = pd.read_csv('../train.csv')
full_data = full_data[(full_data['adr'] < 1000) & (full_data['adr'] > -100)] # remove outliers
full_data = full_data[full_data['is_canceled'] == 0] # only use the uncanceled orders to train

# get the preprocessor and the default training features
preprocessor, feature_spec = datautil.get_the_data_preprocessor()
preprocessor.fit_transform(full_data[feature_spec]) # just for fixing the output dimension of preprocessor

# split the data
train_set, val_set = datautil.split_data_by_date(full_data, val_ratio=VAL_RATIO)
y_train = train_set['adr']
y_val = val_set['adr']
train_set = train_set[feature_spec]
val_set = val_set[feature_spec]
X_train = preprocessor.transform(train_set)
X_val = preprocessor.transform(val_set)

# load model 
nn_reg = keras.models.load_model("adr_regressor.h5")
rf_clf = joblib.load("./rf_cls.model")

# get accuracy score
# score, acc = nn_reg.evaluate(X_val, y_val)
rf_clf.score(X_val, y_val)
# nn_reg.summary()

# loss = nn_reg.evaluate(X_val, y_val, verbose=2)
# print("Model accuracy: {:5.2f}%".format(100 * acc))