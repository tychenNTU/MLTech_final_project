import pandas as pd
import numpy as np
import os
import sys
import datautil
from joblib import dump, load

test_data = pd.read_csv(os.path.join(os.path.dirname(__file__), "test.csv"))
full_data = pd.read_csv(os.path.join(os.path.dirname(__file__), "train.csv"))
full_data = full_data[(full_data['adr'] < 1000) & (full_data['adr'] > -100)] # remove outliers

# get the preprocessor and the default training features
preprocessor, features_spec = datautil.get_the_data_preprocessor()
X_train_full_raw = full_data[features_spec]
preprocessor.fit_transform(X_train_full_raw)
X_test = preprocessor.transform(test_data[features_spec])

# print(list(set(test_data.keys()) - set(features_spec)))

# load model
lightgbm = load("LightGBM_regression/lightgbm.model")
rf_clf = load("classification/lightgbm.model")

# make prediction on the test data
test_data["is_canceled"] = rf_clf.predict(X_test) # predicted is_canceld
test_data["adr"] = lightgbm.predict(X_test)         # predicted adr

# compute the predicted label
revenue_df = datautil.get_revenue_df(test_data)
revenue_df["revenue_label"] = revenue_df["revenue"].apply(lambda revenue: int(revenue/10000))

# save the predicted result
test_no_label_data = pd.read_csv("test_nolabel.csv")
test_no_label_data["label"] = np.array(revenue_df["revenue_label"])
test_no_label_data.to_csv("predicted_label.csv", index=False)

