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
all_data["agent"].replace(np.nan, 0, inplace=True)
# store the columns we added
added_num_features = []
added_cat_features = []

# add arrival_date column
all_data = datautil.get_arrival_date_column(all_data)

# add some features from old features
all_data = datautil.get_weekday_col(all_data)
added_cat_features.append("weekday")

# split the data
train_data_raw, val_data_raw  = datautil.split_data_by_date(all_data, val_ratio=0.2, add_arrival_date=True)

train_data_raw.to_csv("./clf_train_data.csv", index=False)
val_data_raw.to_csv("./clf_val_data.csv", index=False)
added_features = {"added_num_features": added_num_features, "added_cat_features": added_cat_features}
with open("./clf_added_features", "wb") as fd:
    pickle.dump(added_features, fd)