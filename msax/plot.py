import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
import seaborn as sns

from msax.msax import paa

from mpl_toolkits.mplot3d import Axes3D


def ts_with_hist(x, fig=None, bins=50):
    """
    Visualizes the input x time series with its histogram.

    :param x: Input array
    :param fig: Optional, matplotlib figure
    :param bins: Number of bins on histogram
    :return:
    """
    if fig is None:
        fig = plt.figure(figsize=(18, 6))
    gridsize = (1, 3)
    ax1 = plt.subplot2grid(gridsize, (0, 0), colspan=2, rowspan=1, fig=fig)
    ax2 = plt.subplot2grid(gridsize, (0, 2), fig=fig)
    ax1.plot(x)
    ax2.hist(x, bins=bins)

    return fig, ax1, ax2


def plot_paa(x, w, fig=None):
    """
    Visualizes the input x time series with its PAA transformed version.

    :param x: Input array
    :param w: window size parameter for PAA transformation
    :param fig: Optional, matplotlib figure
    :return:
    """
    if fig is None:
        fig = plt.figure(figsize=(18, 6))
    x_paa = paa(x, w)

    ax = fig.subplots(nrows=1, ncols=1)
    ax.plot(x, c='blue', label='Original')
    ax.plot(np.repeat(x_paa, w), c='orange', label='PAA')
    ax.legend()

    return fig, ax


def plot_2d_error_surface(err_surface, fig=None):
    """
    Visualizes the input ErrorSurface in 2D.

    :param err_surface: Input error surface. Must be an instance of ErrorSurface (msax.error.ErrorSurface)
    :type err_surface: msax.error.ErrorSurface
    :param fig: Optional, matplotlib figure
    :return:
    """
    if fig is None:
        fig = plt.figure(figsize=(18, 12))

    err_surface_vals = err_surface.values

    ax = fig.add_subplot(111)
    sns.heatmap(err_surface_vals, xticklabels=err_surface.alphabets, yticklabels=err_surface.windows, ax=ax)
    ax.set_ylabel('window size')
    ax.set_xlabel('alphabet size')
    ax.set_title('Cost of SAX')
    return fig, ax


def plot_3d_error_surface(err_surface, ax=None, title=None):
    """
    Visualizes the input ErrorSurface in 3D.

    :param err_surface: Input error surface
    :type err_surface: msax.error.ErrorSurface
    :param title: Optional. The title of the figure
    :param ax: Optional. matplotlib axes.
    :return:
    """
    if ax is None:
        fig = plt.figure(figsize=(18, 12))
        ax = fig.add_subplot(1, 1, 1, projection='3d')

    from msax.error import ErrorSurface
    window_sizes = err_surface.windows
    alphabet_sizes = err_surface.alphabets
    if isinstance(err_surface, ErrorSurface):
        err_surface = err_surface.values

    x = np.repeat(window_sizes, len(alphabet_sizes)).astype(float)
    y = np.tile(alphabet_sizes, len(window_sizes)).astype(float)
    z = np.ravel(err_surface)

    plt.rcParams.update({'font.size': 20})
    triang = mtri.Triangulation(y, x)
    ax.plot_trisurf(triang, z, cmap='viridis', vmin=np.nanmin(z), vmax=np.nanmax(z))
    ax.set_xlabel('alphabet size')
    ax.set_ylabel('window size')
    ax.set_zlabel('error')

    ax.xaxis.labelpad = 18
    ax.yaxis.labelpad = 18
    ax.zaxis.labelpad = 18

    if not title is None:
        ax.set_title(title)

    plt.rcParams.update({'font.size': 12})

    return ax
