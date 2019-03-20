import numpy as np

from scipy import linalg
from scipy.special import erfinv


break_point_cache = {}


def breakpoints(a):
    """
    Generates breakpoints for 'a' alphabet size.

    :param a: alphabet size
    :type a: int
    :return: List of breakpoint values.
    :rtype: list
    """
    bps = break_point_cache.get(a, None)
    if bps is None:
        bps = [np.sqrt(2)* erfinv(2*(j / a) - 1) for j in range(1, a)]
        break_point_cache[a] = bps
    return bps


def symbol_values(a):
    """
    Returns with the symbol values of the alphabet size.

    :param a: alphabet size
    :type a: int
    :return: List of symbol values
    :rtype: np.ndarray
    """
    bps = breakpoints(a)

    vals = [bps[0]]
    vals.extend([(bps[x] + bps[x - 1]) / 2 for x in range(1,a-1)])
    vals.append(bps[-1])
    return np.array(vals)


def normalize(x):
    """
    Normalize the input vector to zero mean and 1 variance.

    :param x: Input array
    :type x: np.ndarray
    :return: The normalized version of the input vector.
    :rtype: np.ndarray
    """
    return (x - np.mean(x))  / np.std(x)


def paa(x, w):
    """
    Applying PAA on input vector with w window size.

    :param x: input vector
    :type x: np.ndarray
    :param w: window size
    :type w: int
    :return: The PAA transformed version of the input vector
    :rtype: np.ndarray
    """
    win_no = np.int32(np.ceil(x.size / w))
    # The trick: windowing can be achieved by reshaping the original array. It adds nan padding to last window if it is
    # necessary. The nan padding will not be taken into account in mean calculation.
    if x.size % w == 0:
        y = x.reshape(win_no, w)
    else:
        y = np.pad(x, (0, w - x.size % w), 'constant', constant_values=np.nan).reshape(win_no, w)
    return np.nanmean(y, axis=1)


def alphabetize(paa_x, a):
    """
    Alphabetizes the input vector based on 'a' alphabet size.

    :param paa_x: input vector
    :type paa_x: np.ndarray
    :param a: alphabet size
    :type a: int
    :return: Symbol indices' list. A symbol index can be interpreted as an alphabet index.
    :rtype: np.ndarray
    """
    return  np.searchsorted(breakpoints(a), paa_x)


def sax(x, w, a, normalized=False):
    """
    Executes the original SAX transformation.

    :param w: window size
    :type w: int
    :param a: alphabet size
    :type a: int
    :param x: The time series
    :type x: np.ndarray
    :param normalized: Set True if the x input time series is already normalized. Otherwise, leave it False.
    :type normalized: bool
    :return: The symbolic representation of the input time series. The result contains an array of symbol indices.
    :rtype: np.ndarray
    """
    if not normalized:
        return alphabetize(paa(normalize(x),w), a)
    else:
        return alphabetize(paa(x,w), a)


def independence_transform(x):
    """
    Independence transformation for correlated multivariate time series.

    :param x: Matrix of the multivariate time series
    :type x: np.ndarray
    :return: Matrix of the uncorrelated and normalized multivariate time series
    :rtype: np.ndarray
    """
    u, s, vh = linalg.svd(np.cov(x))
    s_mx = linalg.diagsvd(s, len(s), len(s))
    s_inv = linalg.inv(np.sqrt(s_mx))
    trans_mx = np.matmul(s_inv, u)
    transformed = np.matmul(trans_mx, x)

    # already has std = 1. last line create near zero mean
    return [transformed[i] - np.mean(transformed[i]) for i in range(len(transformed))]


def is_independent(x, corr_threshold):
    """
    Investigates the independency between the x's variables based on the provided corr_threshold parameter. It returns
    with True if all pairwise correlation coefficient is below the threshold, otherwise returns False.

    :param x: Matrix of the multivariate time series
    :type x: np.ndarray
    :param corr_threshold: correlation threshold
    :type corr_threshold: float
    :return: True of the variables are independent. Otherwise, False.
    :rtype: bool
    """
    c = np.corrcoef(x)
    np.fill_diagonal(c, 0)
    return np.all(np.abs(c.ravel()) <= corr_threshold)


def msax(x, w, a, corr_threshold=0.1):
    """
    Executes the MSAX transformation on input matrix of a multivariate time series.

    :param x: Matrix of the multivariate time series
    :param w: window size
    :param a: alphabet size
    :param corr_threshold: correlation threshold
    :return: The symbolic representation of matrix. The result contains an array of symbol indices.
    :rtype: np.ndarray
    """

    if len(x) == 1:
        return sax(x[0], w, a)

    if not is_independent(x, corr_threshold):
        x = independence_transform(x)
    else:
        # TODO: ha függetlenek normalizálni kell
        raise NotImplementedError

    xx = sax(x[0], w, a)
    for idx in range(1, len(x)):
        xx += sax(x[idx], w, a, normalized=True) * np.power(a, idx)

    return xx


def symbol_dist(x, y, bps):
    """
    Calculates the distance between x and y symbol indices.

    :param x: x symbol index
    :type x: int
    :param y: y symbol index
    :type y: int
    :param bps: Breakpoint of the alphabet
    :return:
    """
    big, small = (x, y) if x >= y else (y, x)
    if big == small:
        return 0.0
    return bps[big - 1] - bps[small]


def mindist(x, y, a):
    """
    Calculates the distance between two SAX representation. (Equivalent with the MINDIST calculation)

    :param x: List of symbol indices
    :type x: np.ndarray
    :param y: List of symbol indices
    :type y: np.ndarray
    :param a: alphabet size
    :type a: int
    :return: Distance of two SAX representation
    :rtype float
    """
    bps = breakpoints(a)
    ds = [symbol_dist(x[i], y[i], bps) for i in range(len(x))]

    return np.sqrt(x) * np.sqrt(np.sum(np.power(ds,2)))


# TODO: ellenőrizni, hogy tényleg jó-e az implementáció!
def dist(x, y, a, dims=1):
    """
    Calculates distance between two MSAX representation
    :param x:
    :param y:
    :param a:
    :param dims:
    :return:
    """
    if x.size != y.size:
        raise ValueError("x and y vectors' length are not equal")

    if dims == 1:
        return mindist(x, y, a)

    dim_dists = []
    # decomposition of time series along the dimensions
    for dim in range(dims - 1, 0, -1):
        dim_pow = np.power(a, dim)
        # x
        x_dim = np.floor(x / dim_pow)
        x = np.remainder(x, dim_pow)
        # y
        y_dim = np.floor(y / dim_pow)
        y = np.remainder(y, dim_pow)
        # distance calculation along the current dimension
        dim_dists.append(np.power(mindist(x_dim, y_dim, a), 2))

    return np.mean(np.sqrt(np.sum(np.array(dim_dists), axis=0)))