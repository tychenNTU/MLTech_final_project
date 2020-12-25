# common:
import pandas as pd
import numpy as np
import sys
# for ML:
import tensorflow.keras as keras
DATA_UTIL_PATH = "../"
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

# generate the regression model using the default hyperparameters
nn_reg = build_model(input_shape=X_train.shape[1:])

# train the model and save the result
history = nn_reg.fit(X_train, y_train, epochs=2000, validation_data=(X_val, y_val),
           callbacks=[keras.callbacks.EarlyStopping(patience=30, restore_best_weights=True)])
nn_reg.save("./adr_regressor.h5")
pd.DataFrame(history.history).to_csv("nn_training_history.csv", index=False)
