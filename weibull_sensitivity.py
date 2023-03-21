from sklearn.model_selection import train_test_split
from joblib import Parallel, delayed
from scipy.stats import loguniform
from pyDOE import lhs

from logistic_functions import *


def sample_weibull(ratio, dec, sel, noise, type="gaussian"):
    """Get accuracy for a given ratio and shift on two Weibull distribution
    using the logistic regression model with either the geometric or
    arithmetic approach.
    If the optimization fails, return -1 for the ratio, shift and accuracy.
    It create a dataset of 5000 samples of size 121 from each Weibull distribution and
    then concatenate them. The labels are 1 for the first distribution and 0
    for the second. The dataset is then split into training and testing set

    Parameters
    ----------
    ratio : float
        Ratio between the two shapes of the Weibull distribution.
    dec : float
        Shift between the two Weibull distributions.
    sel : str
        Select the approach to use. Either "geometric" or "arithmetic".
    noise : float
        Standard deviation of the noise to add to the Weibull distribution.
    type : str
        Type of noise to add to the Weibull distribution. Either "gaussian" or "speckle".

    Returns
    -------
    tuple
        Tuple containing the ratio, shift and accuracy.
    """
    k = 2
    win_size = 11
    m = np.random.weibull(k * ratio, (5000, win_size**2))
    n = np.random.weibull(k, (5000, win_size**2)) + dec

    if noise == -1:
        random_noise = np.abs(np.random.normal(0, noise, (5000, win_size**2)))
        if type == "gaussian":
            m = m + random_noise
            n = n + random_noise
        elif type == "speckle":
            m = m * random_noise + m
            n = n * random_noise + n

    x = np.concatenate([m, n])
    y = np.concatenate([np.ones(5000), np.zeros(5000)])

    X_train, X_test, Y_train, Y_test = train_test_split(
        x, y, test_size=0.2, random_state=42
    )
    Y_train = Y_train.reshape(1, -1).T
    Y_test = Y_test.reshape(1, -1).T
    w = np.random.rand(X_train.shape[1], 1) * 5e-5
    b = 0.8 * np.ones((1, 1))

    try:
        weights, bias = LR_minimize(X_train, Y_train, w, b, sel, method="l-bfgs-b")
        accuracy_, _ = predict(X_test, Y_test, weights, bias, sel)
        return ratio, dec, accuracy_
    except:
        return -1, -1, -1


def sample_lhs(n, lower=1e-4, upper=1e2):
    """Sample the space of the ratio and shift using the Latin Hypercube Sampling

    Parameters
    ----------
    n : int
        Number of samples to generate
    lower : float, optional
        Lower bound, by default 1e-4
    upper : float, optional
        Upper bound, by default 1e2

    Returns
    -------
    numpy array
        Array containing the log-uniformly distributed of the samples of the latin hypercube
    """
    lhs_val = lhs(n=2, samples=n, criterion="maximin", iterations=100)
    lhs_val[:, [0, 1]] = loguniform(lower, upper).ppf(lhs_val[:, [0, 1]])
    return lhs_val


if __name__ == "__main__":

    output = "../data/"
    makedirs(output, exist_ok=True)

    seed = 42
    np.random.seed(seed)

    hypercube_L = sample_lhs(300, lower=1e-4, upper=1e2)
    ratio, shift = hypercube_L[:, 0], hypercube_L[:, 1]
    print(hypercube_L.shape)

    results = {}
    results["geometric"] = Parallel(n_jobs=-1)(
        delayed(sample_weibull)(r, d, "geometric", -1) for r in ratio for d in shift
    )
    results["arithmetic"] = Parallel(n_jobs=-1)(
        delayed(sample_weibull)(r, d, "arithmetic", -1) for r in ratio for d in shift
    )
    dump_pkl(results, join(output, "weibull_sampling_space.pkl"))
