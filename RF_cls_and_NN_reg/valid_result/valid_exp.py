# common:
import pandas as pd
import numpy as np
import sys
# for ML:
import tensorflow.keras as keras
DATA_UTIL_PATH = "../../"
sys.path.append(DATA_UTIL_PATH)
import datautil

VAL_RATIO = 0.25 # tatio for validation data

# this function just build simple NN model
def build_model(input_shape, n_hidden=2, n_neurons=30, learning_rate=1e-3):
    model = keras.models.Sequential()
    model.add(keras.layers.InputLayer(input_shape=input_shape))
    for _layer in range(n_hidden):
        #model.add(keras.layers.BatchNormalization())
        #model.add(keras.layers.Dropout(rate=0.2))
        model.add(keras.layers.Dense(n_neurons, activation="relu"))
    model.add(keras.layers.Dense(1))
    optimizer = keras.optimizers.SGD(learning_rate=learning_rate, clipnorm=1)
    model.compile(loss="mse", optimizer=optimizer)
    return model

full_data = pd.read_csv('../../train.csv')
full_data = full_data[(full_data['adr'] < 1000) & (full_data['adr'] > -100)] # remove outliers
full_data = full_data[full_data['is_canceled'] == 0] # only use the uncanceled orders to train

# get the preprocessor and the default training features
preprocessor, feature_spec = datautil.get_the_data_preprocessor()
preprocessor.fit_transform(full_data[feature_spec]) # just for fixing the output dimension of preprocessor

# split the data
train_set, val_set = datautil.split_data_by_date(full_data, val_ratio=0.2)
y_train = train_set['adr']
y_val = val_set['adr']
train_set = train_set[feature_spec]
val_set = val_set[feature_spec]
X_train = preprocessor.transform(train_set)
X_val = preprocessor.transform(val_set)

reg_model = build_model(input_shape=X_train.shape[1:])
history = reg_model.fit(X_train, y_train, epochs=2000, validation_data=(X_val, y_val),
              callbacks=[keras.callbacks.EarlyStopping(patience=30, restore_best_weights=True)])
reg_model.save("./adr_regressor.h5")
pd.DataFrame(history.history).to_csv("nn_training_history.csv", index=False)