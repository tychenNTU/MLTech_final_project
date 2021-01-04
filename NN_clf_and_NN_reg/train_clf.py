# common:
import pandas as pd
import numpy as np
import sys
import os
import shutil
import datetime
import pickle
# for ML:
from sklearn.preprocessing import OneHotEncoder
import tensorflow.keras as keras
import tensorflow as tf
DATA_UTIL_PATH = "../"
sys.path.append(DATA_UTIL_PATH)
import datautil

# load data
train_data_raw = pd.read_csv("./clf_train_data.csv")
val_data_raw = pd.read_csv("./clf_val_data.csv")
with open("clf_added_features", "rb") as fd:
    added_features = pickle.load(fd)

added_num_features = added_features["added_num_features"]
added_cat_features = added_features["added_cat_features"]

num_features = ["lead_time","arrival_date_week_number","arrival_date_day_of_month",
                "stays_in_weekend_nights","stays_in_week_nights","adults","children",
                "babies","is_repeated_guest", "previous_cancellations",
                "previous_bookings_not_canceled",
                "required_car_parking_spaces", "total_of_special_requests"] \
                + added_num_features 

cat_features = ["hotel","arrival_date_month","meal","market_segment","agent",
                "distribution_channel","reserved_room_type","deposit_type","customer_type"] \
                + added_cat_features

# get the preprocessor and the default training features
preprocessor, features_spec = datautil.get_the_data_preprocessor(num_features, cat_features)

# split data into input and label
X_train = train_data_raw[features_spec]
y_train = np.array(train_data_raw["is_canceled"])
X_val = val_data_raw[features_spec]
y_val = np.array(val_data_raw["is_canceled"])


X_train = preprocessor.fit_transform(X_train)
X_val = preprocessor.transform(X_val)
binary_encoder = OneHotEncoder(sparse=False)
#y_train = binary_encoder.fit_transform(y_train.reshape(-1, 1))
#y_val = binary_encoder.transform(y_val.reshape(-1, 1))

model = keras.models.Sequential([
    keras.layers.Dense(100, activation="relu", input_shape=X_train.shape[1:]),
    keras.layers.Dense(100, activation="relu"),
    keras.layers.Dense(1, activation="sigmoid")
])
model.compile(loss="binary_crossentropy", optimizer=keras.optimizers.SGD(lr=1e-4), metrics=["accuracy"])
log_path = "./clf_log/" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
os.mkdir(log_path)
history_path = log_path + "/history.csv"
model_path = log_path + "/model.h5"
data_code_path = log_path + "/data.py"
train_code_path = log_path + "/train.py"
shutil.copy("./data_for_clf.py", data_code_path)
shutil.copy("./train_clf.py", train_code_path)
history = model.fit(X_train, y_train, validation_data=(X_val, y_val),
                    epochs=2000, callbacks=[keras.callbacks.EarlyStopping(patience=100, restore_best_weights=True)])
model.save(model_path)
pd.DataFrame(history.history).to_csv(history_path, index=False)