from utils import *
from sklearn.metrics import accuracy_score
from scipy.optimize import minimize
from geometric_functions import *
from arithmetic_functions import *


def forward(x, w, b, s):
    """Forward propagation for the logistic regression model in arthmetic or geometric approach

    Parameters
    ----------
    x : numpy array
        Input data
    w : numpy array
        Weights
    b : float
        Bias
    s : string
        Approach, either 'arithmetic' or 'geometric'

    Returns
    -------
    ye : numpy array
        Predicted values
    """
    if s == "arithmetic":
        ye = LR_arth_forward(x, w, b)
    elif s == "geometric":
        ye = LR_geom_forward(x, w, b)
    else:
        print("Error")
        ye = None
    return ye


def backward(x, y, w, b, ye, m, s):
    """Backward propagation for the logistic regression model in arthmetic or geometric approach

    Parameters
    ----------
    x : numpy array
        Input data
    y : numpy array
        True values
    ye : numpy array
        Predicted values
    m : int
        Number of samples
    s : string
        Approach, either 'arithmetic' or 'geometric'

    Returns
    -------
    dw : numpy array
        Gradient of the weights
    db : float
        Gradient of the bias
    """
    if s == "arithmetic":
        g = LR_arth_backward(x, y, ye, m)
    elif s == "geometric":
        g = LR_geom_backward(x, w, b, y, ye, m)
    else:
        print("Error")
        g = None
    return g


def predict(X, Y, w, b, selection):
    """Predict the values of the test set

    Parameters
    ----------
    X : numpy array
        Input data
    Y : numpy array
        True values
    w : numpy array
        Weights estimated
    b : float
        Bias estimated
    selection : string
        Approach, either 'arithmetic' or 'geometric'

    Returns
    -------
    accuracy_ : float
        Accuracy of the model
    ye : numpy array
        Predicted values
    """
    if selection == "arithmetic":
        ye = LR_arth_forward(X, w, b)
    elif selection == "geometric":
        ye = LR_geom_forward(X, w, b)
    Y_prediction_test = seuillage(ye)
    accuracy_ = 100 * accuracy_score(Y.reshape(-1), Y_prediction_test)
    return accuracy_, ye


def seuillage(h):
    """Seuillage of the output of the logistic regression model

    Parameters
    ----------
    h : numpy array
        Output of the logistic regression model

    Returns
    -------
    numpy array
        Predicted values (0 or 1) with the threshold 0.5
    """
    return np.where(h > 0.5, 1, 0).flatten()


def cost_function(Y, Ye, m):
    """Cost function for the logistic regression model in arthmetic or geometric approach.
    Use the cross-entropy loss function.

    Parameters
    ----------
    Y : numpy array
        True values
    Ye : numpy array
        Predicted values
    m : int
        Number of samples

    Returns
    -------
    cost : float
        Cost of the model (cross-entropy loss function)
    """
    Yis1 = Y == 1
    cost = -(np.log(Ye[Yis1]).sum() + np.log(1 - Ye[~Yis1]).sum()) / m
    return cost


def prepare_data(wb_0, args):
    """Prepare the data for the minimization function

    Parameters
    ----------
    wb_0 : numpy array
        Weights and bias concatenated
    args : tuple
        Tuple of the data (x, y, w_shape, w_size, b_shape, m, s)

    Returns
    -------
    x : numpy array
        Input data
    y : numpy array
        True values
    w : numpy array
        Weights
    b : float
        Bias
    m : int
        Number of samples
    s : string
        Approach, either 'arithmetic' or 'geometric'
    """
    x, y, w_shape, w_size, b_shape, m, s = args
    w = wb_0[:w_size].reshape(w_shape)
    b = wb_0[w_size:].reshape(b_shape)
    return x, y, w, b, m, s


def f(wb_0, *args):
    """Cost function for the minimization function and forward propagation

    Parameters
    ----------
    wb_0 : numpy array
        Weights and bias concatenated

    Returns
    -------
    cost : float
        Cost of the model (cross-entropy loss function)
    """
    global cost
    x, y, w, b, m, s = prepare_data(wb_0, args)
    ye = forward(x, w, b, s)
    cost = cost_function(y, ye, m)
    return cost


def f_prime(wb_0, *args):
    """Gradient of the cost function for the minimization function and backward propagation

    Parameters
    ----------
    wb_0 :  numpy array
        Weights and bias concatenated

    Returns
    -------
    g : numpy array
        Gradient of the weights and bias concatenated
    """
    x, y, w, b, m, s = prepare_data(wb_0, args)
    ye = forward(x, w, b, s)
    g = backward(x, y, w, b, ye, m, s)
    return g


def LR_minimize(X, Y, w, b, selection, method="CG"):
    """Minimization function for the logistic regression model in arthmetic or geometric approach

    Parameters
    ----------
    X : numpy array
        Input data for the training set (m, n) with m the number of samples and n the number of features
    Y : numpy array
        True values for the training set (m, 1)
    w : numpy array
        Weights (n, 1) initialized
    b : float
        Bias initialized
    selection : string
        Approach, either 'arithmetic' or 'geometric'
    method : str, optional
        _description_, by default "CG"

    Returns
    -------
    w : numpy array
        Weights estimated
    b : float
        Bias estimated
    """

    def callbackF(_):
        print(f"\rCost: {cost:.8e}", end="")

    param = (X, Y, w.shape, w.size, b.shape, X.shape[0], selection)
    x0 = np.concatenate([w.flatten(), b.flatten()])
    t = time.time()
    out = minimize(f, x0, args=param, jac=f_prime, method=method, callback=callbackF)
    print(f"\nTime: {time.time()-t:.2f}")

    if out.success:
        w = out.x[: w.size].reshape(w.shape)
        b = out.x[w.size :].reshape(b.shape)
        accuracy_, _ = predict(X, Y, w, b, selection)
        print("\n", out.message)
        print(
            "iterations : ",
            out.nit,
            "| cost : ",
            out.fun,
            "| accuracy train : ",
            accuracy_,
        )
        return w, b

    else:
        print(out.message)
        return None
