import numpy as np
from sklearn.metrics import mean_squared_error

def rmse(y_pred, y_true):
    return np.sqrt(np.mean((y_pred - y_true) ** 2))


def mse(y_pred, y_true):
    return mean_squared_error(y_pred, y_true)


y_pred = np.array([1, 2, 4, 4, 5])
y_true = np.array([1, 1, 1, 1, 1])

print(rmse(y_pred, y_true))
print(mse(y_pred, y_true))
