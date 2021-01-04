# common:
import pandas as pd
import numpy as np
import sys
from datetime import datetime
import pickle
# for ML:
import tensorflow.keras as keras
import tensorflow as tf
DATA_UTIL_PATH = "../"
sys.path.append(DATA_UTIL_PATH)
import datautil
    
all_data_raw = pd.read_csv("../train.csv")
all_data = all_data_raw[(all_data_raw['adr'] < 1000) & (all_data_raw['adr'] > -1000)] # remove outliers

# store the columns we added
added_num_features = []
added_cat_features = []

# add arrival_date column
all_data = datautil.get_arrival_date_column(all_data)

# add some features from old features
all_data = datautil.get_date_difference_col(all_data)
added_num_features.append("date_diff")
all_data = datautil.get_weekday_col(all_data)
added_cat_features.append("weekday")

# split the data
train_data_raw, val_data_raw  = datautil.split_data_by_date(all_data, val_ratio=0.2, rand_seed=87,add_arrival_date=True)

# add some groupby features for training and validation data
group_by_num_features = []
features_to_group_adr = ["agent"]
features_to_group_cancel_rate = ["agent", "arrival_date_month"]
for feature in features_to_group_adr:
    train_data_raw, avg_target_val_dict = datautil.add_group_by_mean_col(train_data_raw, feature, "adr", group_na=True)
    val_data_raw, _ = datautil.add_group_by_mean_col(val_data_raw, feature, "adr", avg_target_val_dict=avg_target_val_dict, group_na=True)
for feature in features_to_group_cancel_rate:
    train_data_raw, avg_target_val_dict = datautil.add_group_by_mean_col(train_data_raw, feature, "is_canceled", group_na=True)
    val_data_raw, _ = datautil.add_group_by_mean_col(val_data_raw, feature, "is_canceled", avg_target_val_dict=avg_target_val_dict, group_na=True)
group_by_num_features += [by_col + "_avg_" + "adr" for by_col in features_to_group_adr] + \
                   [by_col + "_avg_" + "is_canceled" for by_col in features_to_group_cancel_rate]
added_num_features += group_by_num_features

train_data_raw.to_csv("./reg_train_data.csv", index=False)
val_data_raw.to_csv("./reg_val_data.csv", index=False)
added_features = {"added_num_features": added_num_features, "added_cat_features": added_cat_features}
with open("./reg_added_features", "wb") as fd:
    pickle.dump(added_features, fd)