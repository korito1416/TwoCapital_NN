import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.stats import norm
import warnings
import time
import pickle

# from numba import njit



def finiteDiff_4D(data, dim, order, dlt, cap=None):
    # compute the central difference derivatives for given input and dimensions
    res = np.zeros(data.shape)
    l = len(data.shape)
    if l == 4:
        if order == 1:  # first order derivatives
            if dim == 0:  # to first dimension
                res[1:-1, :, :, :] = (1 / (2 * dlt)) * (data[2:, :, :, :] - data[:-2, :, :, :])
                res[-1, :, :, :] = (1 / dlt) * (data[-1, :, :, :] - data[-2, :, :, :])
                res[0, :, :, :] = (1 / dlt) * (data[1, :, :, :] - data[0, :, :, :])

            elif dim == 1:  # to second dimension
                res[:, 1:-1, :, :] = (1 / (2 * dlt)) * (data[:, 2:, :, :] - data[:, :-2, :, :])
                res[:, -1, :, :] = (1 / dlt) * (data[:, -1, :, :] - data[:, -2, :, :])
                res[:, 0, :, :] = (1 / dlt) * (data[:, 1, :, :] - data[:, 0, :, :])

            elif dim == 2:  # to third dimension
                res[:, :, 1:-1, :] = (1 / (2 * dlt)) * (data[:, :, 2:, :] - data[:, :, :-2, :])
                res[:, :, -1, :] = (1 / dlt) * (data[:, :, -1, :] - data[:, :, -2, :])
                res[:, :, 0, :] = (1 / dlt) * (data[:, :, 1, :] - data[:, :, 0, :])

            elif dim == 3:  # to fourth dimension
                res[:, :, :, 1:-1] = (1 / (2 * dlt)) * (data[:, :, :, 2:] - data[:, :, :, :-2])
                res[:, :, :, -1] = (1 / dlt) * (data[:, :, :, -1] - data[:, :, :, -2])
                res[:, :, :, 0] = (1 / dlt) * (data[:, :, :, 1] - data[:, :, :, 0])

            else:
                raise ValueError('wrong dim')

        elif order == 2:
            if dim == 0:
                res[1:-1, :, :, :] = (1 / dlt ** 2) * (data[2:, :, :, :] + data[:-2, :, :, :] - 2 * data[1:-1, :, :, :])
                res[-1, :, :, :] = (1 / dlt ** 2) * (data[-1, :, :, :] + data[-3, :, :, :] - 2 * data[-2, :, :, :])
                res[0, :, :, :] = (1 / dlt ** 2) * (data[2, :, :, :] + data[0, :, :, :] - 2 * data[1, :, :, :])

            elif dim == 1:
                res[:, 1:-1, :, :] = (1 / dlt ** 2) * (data[:, 2:, :, :] + data[:, :-2, :, :] - 2 * data[:, 1:-1, :, :])
                res[:, -1, :, :] = (1 / dlt ** 2) * (data[:, -1, :, :] + data[:, -3, :, :] - 2 * data[:, -2, :, :])
                res[:, 0, :, :] = (1 / dlt ** 2) * (data[:, 2, :, :] + data[:, 0, :, :] - 2 * data[:, 1, :, :])

            elif dim == 2:
                res[:, :, 1:-1, :] = (1 / dlt ** 2) * (data[:, :, 2:, :] + data[:, :, :-2, :] - 2 * data[:, :, 1:-1, :])
                res[:, :, -1, :] = (1 / dlt ** 2) * (data[:, :, -1, :] + data[:, :, -3, :] - 2 * data[:, :, -2, :])
                res[:, :, 0, :] = (1 / dlt ** 2) * (data[:, :, 2, :] + data[:, :, 0, :] - 2 * data[:, :, 1, :])

            elif dim == 3:
                res[:, :, :, 1:-1] = (1 / dlt ** 2) * (data[:, :, :, 2:] + data[:, :, :, :-2] - 2 * data[:, :, :, 1:-1])
                res[:, :, :, -1] = (1 / dlt ** 2) * (data[:, :, :, -1] + data[:, :, :, -3] - 2 * data[:, :, :, -2])
                res[:, :, :, 0] = (1 / dlt ** 2) * (data[:, :, :, 2] + data[:, :, :, 0] - 2 * data[:, :, :, 1])

            else:
                raise ValueError('wrong dim')
        else:
            raise ValueError('wrong order')
    else:
        raise ValueError("Dimension NOT supported")

    if cap is not None:
        res[res < cap] = cap
    return res



def finiteDiff_5D(data, dim, order, dlt, cap=None):
    # compute the central difference derivatives for given input and dimensions
    res = np.zeros(data.shape)
    l = len(data.shape)
    
    if l == 5:
        if order == 1:  # first order derivatives
            if dim == 0:  # to first dimension
                res[1:-1, :, :, :, :] = (1 / (2 * dlt)) * (data[2:, :, :, :, :] - data[:-2, :, :, :, :])
                res[-1, :, :, :, :] = (1 / dlt) * (data[-1, :, :, :, :] - data[-2, :, :, :, :])
                res[0, :, :, :, :] = (1 / dlt) * (data[1, :, :, :, :] - data[0, :, :, :, :])

            elif dim == 1:  # to second dimension
                res[:, 1:-1, :, :, :] = (1 / (2 * dlt)) * (data[:, 2:, :, :, :] - data[:, :-2, :, :, :])
                res[:, -1, :, :, :] = (1 / dlt) * (data[:, -1, :, :, :] - data[:, -2, :, :, :])
                res[:, 0, :, :, :] = (1 / dlt) * (data[:, 1, :, :, :] - data[:, 0, :, :, :])

            elif dim == 2:  # to third dimension
                res[:, :, 1:-1, :, :] = (1 / (2 * dlt)) * (data[:, :, 2:, :, :] - data[:, :, :-2, :, :])
                res[:, :, -1, :, :] = (1 / dlt) * (data[:, :, -1, :, :] - data[:, :, -2, :, :])
                res[:, :, 0, :, :] = (1 / dlt) * (data[:, :, 1, :, :] - data[:, :, 0, :, :])

            elif dim == 3:  # to fourth dimension
                res[:, :, :, 1:-1, :] = (1 / (2 * dlt)) * (data[:, :, :, 2:, :] - data[:, :, :, :-2, :])
                res[:, :, :, -1, :] = (1 / dlt) * (data[:, :, :, -1, :] - data[:, :, :, -2, :])
                res[:, :, :, 0, :] = (1 / dlt) * (data[:, :, :, 1, :] - data[:, :, :, 0, :])

            elif dim == 4:  # to fifth dimension
                res[:, :, :, :, 1:-1] = (1 / (2 * dlt)) * (data[:, :, :, :, 2:] - data[:, :, :, :, :-2])
                res[:, :, :, :, -1] = (1 / dlt) * (data[:, :, :, :, -1] - data[:, :, :, :, -2])
                res[:, :, :, :, 0] = (1 / dlt) * (data[:, :, :, :, 1] - data[:, :, :, :, 0])

            else:
                raise ValueError('wrong dim')

        elif order == 2:  # second order derivatives
            if dim == 0:  # to first dimension
                res[1:-1, :, :, :, :] = (1 / dlt ** 2) * (data[2:, :, :, :, :] + data[:-2, :, :, :, :] - 2 * data[1:-1, :, :, :, :])
                res[-1, :, :, :, :] = (1 / dlt ** 2) * (data[-1, :, :, :, :] + data[-3, :, :, :, :] - 2 * data[-2, :, :, :, :])
                res[0, :, :, :, :] = (1 / dlt ** 2) * (data[2, :, :, :, :] + data[0, :, :, :, :] - 2 * data[1, :, :, :, :])

            elif dim == 1:  # to second dimension
                res[:, 1:-1, :, :, :] = (1 / dlt ** 2) * (data[:, 2:, :, :, :] + data[:, :-2, :, :, :] - 2 * data[:, 1:-1, :, :, :])
                res[:, -1, :, :, :] = (1 / dlt ** 2) * (data[:, -1, :, :, :] + data[:, -3, :, :, :] - 2 * data[:, -2, :, :, :])
                res[:, 0, :, :, :] = (1 / dlt ** 2) * (data[:, 2, :, :, :] + data[:, 0, :, :, :] - 2 * data[:, 1, :, :, :])

            elif dim == 2:  # to third dimension
                res[:, :, 1:-1, :, :] = (1 / dlt ** 2) * (data[:, :, 2:, :, :] + data[:, :, :-2, :, :] - 2 * data[:, :, 1:-1, :, :])
                res[:, :, -1, :, :] = (1 / dlt ** 2) * (data[:, :, -1, :, :] + data[:, :, -3, :, :] - 2 * data[:, :, -2, :, :])
                res[:, :, 0, :, :] = (1 / dlt ** 2) * (data[:, :, 2, :, :] + data[:, :, 0, :, :] - 2 * data[:, :, 1, :, :])

            elif dim == 3:  # to fourth dimension
                res[:, :, :, 1:-1, :] = (1 / dlt ** 2) * (data[:, :, :, 2:, :] + data[:, :, :, :-2, :] - 2 * data[:, :, :, 1:-1, :])
                res[:, :, :, -1, :] = (1 / dlt ** 2) * (data[:, :, :, -1, :] + data[:, :, :, -3, :] - 2 * data[:, :, :, -2, :])
                res[:, :, :, 0, :] = (1 / dlt ** 2) * (data[:, :, :, 2, :] + data[:, :, :, 0, :] - 2 * data[:, :, :, 1, :])

            elif dim == 4:  # to fifth dimension
                res[:, :, :, :, 1:-1] = (1 / dlt ** 2) * (data[:, :, :, :, 2:] + data[:, :, :, :, :-2] - 2 * data[:, :, :, :, 1:-1])
                res[:, :, :, :, -1] = (1 / dlt ** 2) * (data[:, :, :, :, -1] + data[:, :, :, :, -3] - 2 * data[:, :, :, :, -2])
                res[:, :, :, :, 0] = (1 / dlt ** 2) * (data[:, :, :, :, 2] + data[:, :, :, :, 0] - 2 * data[:, :, :, :, 1])

            else:
                raise ValueError('wrong dim')
        else:
            raise ValueError('wrong order')
    else:
        raise ValueError("Dimension NOT supported")

    if cap is not None:
        res[res < cap] = cap
    return res




def finiteDiff_6D(data, dim, order,dlt, cap=None):
    # compute the first order central difference derivatives for 6D input and dimensions
    res = np.zeros(data.shape)
    
    if dim == 0:
        res[1:-1, :, :, :, :, :] = (1 / (2 * dlt)) * (data[2:, :, :, :, :, :] - data[:-2, :, :, :, :, :])
        res[-1, :, :, :, :, :] = (1 / dlt) * (data[-1, :, :, :, :, :] - data[-2, :, :, :, :, :])
        res[0, :, :, :, :, :] = (1 / dlt) * (data[1, :, :, :, :, :] - data[0, :, :, :, :, :])

    elif dim == 1:
        res[:, 1:-1, :, :, :, :] = (1 / (2 * dlt)) * (data[:, 2:, :, :, :, :] - data[:, :-2, :, :, :, :])
        res[:, -1, :, :, :, :] = (1 / dlt) * (data[:, -1, :, :, :, :] - data[:, -2, :, :, :, :])
        res[:, 0, :, :, :, :] = (1 / dlt) * (data[:, 1, :, :, :, :] - data[:, 0, :, :, :, :])

    elif dim == 2:
        res[:, :, 1:-1, :, :, :] = (1 / (2 * dlt)) * (data[:, :, 2:, :, :, :] - data[:, :, :-2, :, :, :])
        res[:, :, -1, :, :, :] = (1 / dlt) * (data[:, :, -1, :, :, :] - data[:, :, -2, :, :, :])
        res[:, :, 0, :, :, :] = (1 / dlt) * (data[:, :, 1, :, :, :] - data[:, :, 0, :, :, :])

    elif dim == 3:
        res[:, :, :, 1:-1, :, :] = (1 / (2 * dlt)) * (data[:, :, :, 2:, :, :] - data[:, :, :, :-2, :, :])
        res[:, :, :, -1, :, :] = (1 / dlt) * (data[:, :, :, -1, :, :] - data[:, :, :, -2, :, :])
        res[:, :, :, 0, :, :] = (1 / dlt) * (data[:, :, :, 1, :, :] - data[:, :, :, 0, :, :])

    elif dim == 4:
        res[:, :, :, :, 1:-1, :] = (1 / (2 * dlt)) * (data[:, :, :, :, 2:, :] - data[:, :, :, :, :-2, :])
        res[:, :, :, :, -1, :] = (1 / dlt) * (data[:, :, :, :, -1, :] - data[:, :, :, :, -2, :])
        res[:, :, :, :, 0, :] = (1 / dlt) * (data[:, :, :, :, 1, :] - data[:, :, :, :, 0, :])

    elif dim == 5:
        res[:, :, :, :, :, 1:-1] = (1 / (2 * dlt)) * (data[:, :, :, :, :, 2:] - data[:, :, :, :, :, :-2])
        res[:, :, :, :, :, -1] = (1 / dlt) * (data[:, :, :, :, :, -1] - data[:, :, :, :, :, -2])
        res[:, :, :, :, :, 0] = (1 / dlt) * (data[:, :, :, :, :, 1] - data[:, :, :, :, :, 0])

    else:
        raise ValueError('wrong dim')

    if cap is not None:
        res[res < cap] = cap
    return res



def finiteDiff_7D(data, dim,order, dlt, cap=None):
    # compute the first order central difference derivatives for 7D input and dimensions
    res = np.zeros(data.shape)
    
    if dim == 0:
        res[1:-1, :, :, :, :, :, :] = (1 / (2 * dlt)) * (data[2:, :, :, :, :, :, :] - data[:-2, :, :, :, :, :, :])
        res[-1, :, :, :, :, :, :] = (1 / dlt) * (data[-1, :, :, :, :, :, :] - data[-2, :, :, :, :, :, :])
        res[0, :, :, :, :, :, :] = (1 / dlt) * (data[1, :, :, :, :, :, :] - data[0, :, :, :, :, :, :])

    elif dim == 1:
        res[:, 1:-1, :, :, :, :, :] = (1 / (2 * dlt)) * (data[:, 2:, :, :, :, :, :] - data[:, :-2, :, :, :, :, :, :])
        res[:, -1, :, :, :, :, :] = (1 / dlt) * (data[:, -1, :, :, :, :, :, :] - data[:, -2, :, :, :, :, :, :])
        res[:, 0, :, :, :, :, :] = (1 / dlt) * (data[:, 1, :, :, :, :, :, :] - data[:, 0, :, :, :, :, :, :])

    elif dim == 2:
        res[:, :, 1:-1, :, :, :, :] = (1 / (2 * dlt)) * (data[:, :, 2:, :, :, :, :] - data[:, :, :-2, :, :, :, :])
        res[:, :, -1, :, :, :, :] = (1 / dlt) * (data[:, :, -1, :, :, :, :] - data[:, :, -2, :, :, :, :])
        res[:, :, 0, :, :, :, :] = (1 / dlt) * (data[:, :, 1, :, :, :, :] - data[:, :, 0, :, :, :, :])

    elif dim == 3:
        res[:, :, :, 1:-1, :, :, :] = (1 / (2 * dlt)) * (data[:, :, :, 2:, :, :, :] - data[:, :, :, :-2, :, :, :])
        res[:, :, :, -1, :, :, :] = (1 / dlt) * (data[:, :, :, -1, :, :, :] - data[:, :, :, -2, :, :, :])
        res[:, :, :, 0, :, :, :] = (1 / dlt) * (data[:, :, :, 1, :, :, :] - data[:, :, :, 0, :, :, :])

    elif dim == 4:
        res[:, :, :, :, 1:-1, :, :] = (1 / (2 * dlt)) * (data[:, :, :, :, 2:, :, :] - data[:, :, :, :, :-2, :, :])
        res[:, :, :, :, -1, :, :] = (1 / dlt) * (data[:, :, :, :, -1, :, :] - data[:, :, :, :, -2, :, :])
        res[:, :, :, :, 0, :, :] = (1 / dlt) * (data[:, :, :, :, 1, :, :] - data[:, :, :, :, 0, :, :])

    elif dim == 5:
        res[:, :, :, :, :, 1:-1, :] = (1 / (2 * dlt)) * (data[:, :, :, :, :, 2:, :] - data[:, :, :, :, :, :-2, :])
        res[:, :, :, :, :, -1, :] = (1 / dlt) * (data[:, :, :, :, :, -1, :] - data[:, :, :, :, :, -2, :])
        res[:, :, :, :, :, 0, :] = (1 / dlt) * (data[:, :, :, :, :, 1, :] - data[:, :, :, :, :, 0, :])

    elif dim == 6:
        res[:, :, :, :, :, :, 1:-1] = (1 / (2 * dlt)) * (data[:, :, :, :, :, :, 2:] - data[:, :, :, :, :, :, :-2])
        res[:, :, :, :, :, :, -1] = (1 / dlt) * (data[:, :, :, :, :, :, -1] - data[:, :, :, :, :, :, -2])
        res[:, :, :, :, :, :, 0] = (1 / dlt) * (data[:, :, :, :, :, :, 1] - data[:, :, :, :, :, :, 0])

    else:
        raise ValueError('wrong dim')

    if cap is not None:
        res[res < cap] = cap
    return res

