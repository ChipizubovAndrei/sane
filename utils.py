import numpy as np

"""
Функция активации ReLU
"""
def ReLU(x):
    for i in range(len(x)):
        if x[i] < 0:
            x[i] = 0

"""
Функция активации SoftMax
"""
def SoftMax(x):
    out = np.zeros(x.shape)
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            out[i, j] = np.exp(x[i, j]) / np.sum(np.exp(x[i]))
    return out