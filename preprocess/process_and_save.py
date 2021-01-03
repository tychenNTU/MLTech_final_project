import sys
DATA_UTIL_PATH = "../"
sys.path.append(DATA_UTIL_PATH)
import datautil
import numpy as np
import pickle as pkl

train_x, train_y = datautil.get_preprocessed_xy()

#to save it
with open("processed_train.pkl", "wb") as f:
    pkl.dump([train_x, train_y], f)

#to load it
with open("processed_train.pkl", "rb") as f:
    train_x, train_y = pkl.load(f)
    print("len of x is " + str(len(train_x)))
