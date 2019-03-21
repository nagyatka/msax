import numpy as np
import pandas as pd

from msax.msax import normalize, paa, alphabetize, symbol_values


def paa_error(norm_x, paa_x, w):
    """
    Calculates the error of PAA.

    :param norm_x: Normalized time series
    :type norm_x: np.ndarray
    :param paa_x: PAA transformed time series
    :type paa_x: np.ndarray
    :param w: window size
    :type w: int
    :return: Value of PAA error
    :rtype: float
    """
    repeated_paa = np.repeat(paa_x, w)
    # it can happen that repeated_paa will be longer than norm_x
    diff = repeated_paa.size - norm_x.size
    if diff != 0:
        repeated_paa = repeated_paa[:-diff]
    return np.sum(np.abs(repeated_paa - norm_x))


def alphabet_error(paa_x, sax_x, a):
    """
    Calculates the error comes from translation to alphabet.

    :param sax_x: SAX representation
    :type sax_x: np.ndarray
    :param paa_x: PAA transformed time series
    :type paa_x: np.ndarray
    :param a: alphabet size
    :type a: int
    :return: Value of symbolic transformation error
    :rtype: float
    """
    bps_vals = symbol_values(a)
    values = np.array([bps_vals[sym] for sym in sax_x])
    return np.sum(np.abs(values - paa_x))


def sax_error(x, a, w, memory_limit, use_inf=False, l_1 = 1.0):
    """
    Calculates the L_1 error with input parameters. If the w < 2, a < 3 or the generated sax does not fit in the memory
    it will return with np.nan. If the use_inf is True, it returns with np.inf instead of np.nan.

    :param x: input time series
    :type x: np.ndarray
    :param a: alphabet size
    :type a: int
    :param w: window size
    :type w: int
    :param memory_limit: The maximum available memory
    :type memory_limit: int
    :param use_inf: If the use_inf is True, it returns with np.inf instead of np.nan
    :type use_inf: bool
    :param l_1: L_1 regularization controller
    :type l_1: float
    :return: The SAX error
    :rtype: float
    """

    if w < 2 or a < 3:
        return np.nan if not use_inf else np.inf

    if (len(x) / w) * np.log2(a) > memory_limit:
        return np.nan if not use_inf else np.inf

    norm_x = normalize(x)
    paa_x = paa(norm_x, w)
    sym_x = alphabetize(paa_x, a)
    return (paa_error(norm_x, paa_x, w) + alphabet_error(paa_x, sym_x, a) * w) + (a + w) * l_1


def error_surface(x_source, alphabet_sizes, window_sizes, m_size, l_1=1.0):
    """
    Generates the error surface by calculating the error at all possible combinations.

    :param x_source: Iterable object which contains input vectors
    :param alphabet_sizes: List of alphabet sizes
    :param window_sizes: List of window sizes
    :param m_size:
    :return:
    """

    # Preparing the surface
    res = np.full((len(window_sizes), len(alphabet_sizes)), np.nan)

    for idx_s, alphabet_s in enumerate(alphabet_sizes):
        for idx_w, window in enumerate(window_sizes):
            res[idx_w][idx_s] = np.mean([sax_error(x, alphabet_s, window, m_size, l_1) for x in x_source])
    return ErrorSurface(res, alphabet_sizes, window_sizes)


class ErrorSurface(object):
    def __init__(self, values, alphabets, windows):
        self.values = values
        self.alphabets = alphabets
        self.windows = windows

    @property
    def min(self):
        """
        :return: Minimum value of error surface.
        :rtype: float
        """
        return self.values[self.min_coord[0]][self.min_coord[1]]

    @property
    def min_coord(self):
        flat_min = np.nanargmin(self.values)
        return int(flat_min / self.values.shape[1]), flat_min % self.values.shape[1]

    @property
    def min_w(self):
        return self.windows[self.min_coord[0]]

    @property
    def min_a(self):
        return self.alphabets[self.min_coord[1]]

    def as_dataframe(self, without_nan=True):
        df_x = np.repeat(self.windows, len(self.alphabets)).astype(float)
        df_y = np.tile(self.alphabets, len(self.windows)).astype(float)
        df_z = np.ravel(self.values)

        df = pd.DataFrame({'window': df_x, 'alphabet': df_y, 'err': df_z})
        if without_nan:
            df = df.dropna()
        return df


    def __str__(self):
        return "ErrorSurface: min_w: {}, min_a: {}, min_value: {}".format(self.min_w, self.min_a, self.min)

    def __repr__(self):
        return self.__str__()


