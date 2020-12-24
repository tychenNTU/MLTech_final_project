import pandas as pd
import numpy as np
import util
from sklearn.externals import joblib
import tensorflow.keras as keras 

test_data = pd.read_csv("../test.csv")


# the followings preprocess the test for NN regressor
nn_train_data = pd.read_csv('../train.csv')
nn_train_data = nn_train_data[(nn_train_data['adr'] < 1000) & (nn_train_data['adr'] > -100)] # remove outliers
nn_train_data = nn_train_data[nn_train_data['is_canceled'] == 0] # only use the uncanceled orders to train

# get the nn_preprocessor and the default training features
nn_preprocessor, nn_features_spec = util.get_the_data_preprocessor()

# transform the test data
X_train_nn = nn_preprocessor.fit_transform(nn_train_data)
X_test_nn = nn_preprocessor.transform(test_data[nn_features_spec])


# the followings preprocess the test for random forest classifier
rf_train_data = pd.read_csv('../train.csv')
rf_train_data = rf_train_data[(rf_train_data['adr'] < 1000) & (rf_train_data['adr'] > -100)]
rf_preprocessor, rf_features_spec = util.get_the_data_preprocessor()

# transform the test data for random forest classifier
X_train_rf = rf_preprocessor.fit_transform(rf_train_data)
X_test_rf = rf_preprocessor.transform(test_data[rf_features_spec])


# load model and make prediction
nn_reg = keras.models.load_model("adr_regressor.h5")
rf_clf = joblib.load("./rf_cls.model")
test_data["is_canceled"] = rf_clf.predict(X_test_rf) # predicted is_canceld
test_data["adr"] = nn_reg.predict(X_test_nn)         # predicted adr

# compute the predicted label
revenue_df = util.get_revenue_df(test_data)
revenue_df["revenue_label"] = revenue_df["revenue"].apply(lambda revenue: int(revenue/10000))

# save the predicted result
test_no_label_data = pd.read_csv("../test_nolabel.csv")
test_no_label_data["label"] = np.array(revenue_df["revenue_label"])
test_no_label_data.to_csv("predicted_label.csv", index=False)