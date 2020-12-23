import numpy as np
def RMSE(y_pred, y_true):
    err = 0
    for i in range(len(y_pred)):
        err += ((y_pred[i] - y_true[i]) ** 2)
    err /= len(y_pred)
    err = np.sqrt(err)
    return err


def MSE(y_pred, y_true):
    err = 0
    for i in range(len(y_pred)):
        err += ((y_pred[i] - y_true[i]) ** 2)
    err /= len(y_pred)
    return err

def MAE(y_pred, y_true):
    err = 0
    for i in range(len(y_pred)):
        err += (np.abs(y_pred[i] - y_true[i]))
    err /= len(y_pred)
    return err

def ME(y_pred, y_true):
    err = 0
    for i in range(len(y_pred)):
        err += y_true[i] - y_pred[i]
    err /= len(y_pred)
    return err
