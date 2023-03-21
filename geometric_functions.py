import numpy as np


def sigmoid_geom(z):
    """Sigmoid function in geometric approach

    Parameters
    ----------
    z : numpy array
        Input data

    Returns
    -------
    A : numpy array
        Sigmoid of the input data
    """
    A = z / (1 + z)
    return A


def LR_geom_forward(x, w, b):
    """Forward propagation for the logistic regression model in geometric approach

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
    z : numpy array
        output of the sigmoid function
    """
    A = np.log(x)
    u = np.exp(np.dot(A, w) + b)
    z = sigmoid_geom(u)
    return z


def LR_geom_backward(x, w, b, y, ye, m):
    """Backward propagation for the logistic regression model in geometric approach

    Parameters
    ----------
    x : numpy array
        Input data
    w : numpy array
        Weights
    b : float
        Bias
    y : numpy array
        True values
    ye : numpy array
        Predicted values
    m : int
        Number of samples

    Returns
    -------
    g : numpy array
        Gradient of the weights and bias concatenated
    """
    A = np.log(x)
    gw = np.dot(A.T, (ye - y)) / m
    gb = np.sum(ye - y) / m
    return np.concatenate([gw.flatten(), gb.flatten()])
