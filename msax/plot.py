import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
import seaborn as sns



def ts_with_hist(x, fig = None, gridsize=(1, 3), bins=50):
    """

    :param x:
    :param fig:
    :param gridsize:
    :param bins:
    :return:
    """
    if fig is None:
        fig = plt.figure(figsize=(18, 6))
    ax1 = plt.subplot2grid(gridsize, (0, 0), colspan=2, rowspan=1, fig=fig)
    ax2 = plt.subplot2grid(gridsize, (0, 2), fig=fig)
    ax1.plot(x)
    ax2.hist(x, bins=bins)

    return fig, ax1, ax2


def plot_paa(orig_x, paa_x, w, fig = None):
    """

    :param orig_x:
    :param paa_x:
    :param w:
    :param fig:
    :return:
    """
    if fig is None:
        fig = plt.figure(figsize=(18, 6))
    ax = fig.subplots(nrows=1, ncols=1)
    ax.plot(orig_x, c='blue', label='Original')
    ax.plot(np.repeat(paa_x, w), c='orange', label='PAA')
    ax.legend()

    return fig, ax


def plot_2d_error_surface(err_surface, alphabets, windows, fig = None):
    """

    :param err_surface:
    :param alphabets:
    :param windows:
    :param fig:
    :return:
    """
    if fig is None:
        fig = plt.figure(figsize=(18, 12))

    from msax.error import ErrorSurface
    if isinstance(err_surface, ErrorSurface):
        err_surface = err_surface.values

    ax = fig.add_subplot(111)
    ax.text()
    sns.heatmap(err_surface, xticklabels=alphabets, yticklabels=windows, ax=ax)
    ax.set_ylabel('window size')
    ax.set_xlabel('alphabet size')
    ax.set_title('Cost of SAX')
    sns.jointplot()
    return fig, ax


def plot_3d_error_surface(err_surface, alphabets, windows, ax=None, title=None):
    """

    :param ax:
    :param err_surface:
    :param alphabets:
    :param windows:
    :return:
    """
    if ax is None:
        fig = plt.figure(figsize=(18, 12))
        ax = fig.add_subplot(1, 1, 1, projection='3d')

    from msax.error import ErrorSurface
    if isinstance(err_surface, ErrorSurface):
        err_surface = err_surface.values

    x = np.repeat(windows, len(alphabets)).astype(float)
    y = np.tile(alphabets, len(windows)).astype(float)
    z = np.ravel(err_surface)

    triang = mtri.Triangulation(y, x)
    ax.plot_trisurf(triang, z, cmap='viridis', vmin=np.nanmin(z), vmax=np.nanmax(z))
    ax.set_xlabel('Alphabet')
    ax.set_ylabel('Windows')
    ax.set_zlabel('Error')

    if not title is None:
        ax.set_title(title)

    return ax
