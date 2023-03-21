import numpy as np


def sigmoid(z):
    """Sigmoid function

    Parameters
    ----------
    z :  numpy array
        Input data

    Returns
    -------
    A : numpy array
        Sigmoid of the input data
    """
    return 1 / (1 + np.exp(-z))


def LR_arth_forward(x, w, b):
    """Forward propagation for the logistic regression model in arthmetic approach

    Parameters
    ----------
    x : numpy array
        Input data
    w : numpy array
        Weights
    b : float
        Bias

    Returns
    -------
    ye : numpy array
        Predicted values
    """
    z = np.dot(x, w) + b
    A = sigmoid(z)
    return A


def LR_arth_backward(x, y, ye, m):
    """Backward propagation for the logistic regression model in arthmetic approach

    Parameters
    ----------
    x :  numpy array
        Input data
    y :  numpy array
        True values
    ye : numpy array
        Predicted values
    m :  int
        Number of samples

    Returns
    -------
    dw : numpy array
        Gradient of the weights
    db : float
        Gradient of the bias
    """
    gw = np.dot(x.T, (ye - y)) / m
    gb = np.sum(ye - y) / m
    return np.concatenate([gw.flatten(), gb.flatten()])
