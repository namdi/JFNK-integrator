# This file was written by Dr. Namdi Brandon
# ORCID: 0000-0001-7050-1538
# June 29, 2018

"""
This module contains information about plotting.
"""

# ===============================================
# import
# ===============================================
import matplotlib.pylab as plt
import numpy as np
import numpy.linalg as LA

import integrator

# ===============================================
# function
# ===============================================
def corrections(d_sdc_norm, d_jfnk_norm, y_spect_norm, do_rerr, labels=None, do_legend=False):

    """
    This function plots the magnitude of the correction from the SDC method and JFNK method for \
    each iteration.

    :param numpy.ndarray d_sdc_norm: the maximum norm of the approximation \
    :math:`\\| \\delta^{[k]}_{sdc} \\|` from spectral deferred corrections (SDC) method
    :param numpy.ndarray d_jfnk_norm: the maximum norm of the approximation \
    :math:`\\| \\delta^{[k]}_{jfnk} \\|` from the Jacobian-Free Newton-Krylov (JFNK) method
    :param numpy.ndarray y_spect_norm: the norm of the spectral solution :math:`\\| y_{spect} \\|` \
    used to scale the corrections for relative error
    :param bool do_rerr: the is flag indicates whether the relative error (if True) or the \
    absolute error (if False) should be plotted
    :param list labels: these are the function
    :param bool do_legend: this flag indicates whether or not a legend will be shown

    :return: None
    """
    #
    # plot the corrections in SDC and JFNK
    #

    # plot d(k) / y_spect
    data = (d_sdc_norm, d_jfnk_norm)

    ylabel = 'log 10 Norm'

    if do_rerr:
        data = [x / y_spect_norm for x in data]
        main_title = 'Relative Corrections'
    else:
        main_title = 'Absolute Corrections'

    titles = ['SDC', 'JFNK']
    xlabel = 'Iteration'

    plot_errors(data, titles, labels, do_legend=do_legend, main_title=main_title, xlabel=xlabel, ylabel=ylabel, ls='-o')

    return

def plot_correction_time_steps(D, Y, do_save=False, fpath=None, do_close=False):

    """
    This function plots the magnitude of the corrections (both absolute and relative) \
    :math:`\\| \\delta \\|` vs iteration for ech time step.

    :param list D: the corrections over the simulation. List of length number of time steps, \
    containing a list of length number of iterations for the given time step, of the corrections \
    for the iteration. The corrections are of dimensions (number of time nodes, size of the problem)
    :param list Y: the corresponding approximations to the given correction.
    :param bool do_save: indicating whether or not to save the data
    :param str fpath: the file path in which to save the data
    :param bool do_close: a flag indicating whether or not to close the plots

    :return: None
    """

    # for each time step
    for k in range( len(D) ):

        # the list of the values per iteration
        d = D[k]
        y = Y[k]

        # for each iteration
        x = np.array( [ LA.norm(x, axis=0).max() for x in d ] )
        u = np.array( [ integrator.relative_norm(dd, yy) for dd, yy in zip(d, y) ] )

        # plot for each time step
        plt.figure(k)
        plt.title('step: ' + str(k))

        # plot the absolute correction
        plt.plot(range(len(x)), np.log10(x), '-o', label='|d|')

        # plot the relative correction
        plt.plot(range(len(u)), np.log10(u), '-o', label='|d|/|y|')

        plt.xlabel('iteration')
        plt.ylabel('log10(|d|)')
        plt.legend(loc='best')

        # save the figure
        if do_save:
            fname = fpath + ('\\step%.3d.pdf' % k)
            plt.savefig(fname, bbox_inches='tight')

        # close the figure
        if do_close:
            plt.close()

    return

def plot_errors(data, titles, labels, do_legend=False, main_title='', xlabel='', ylabel='', ls='-'):

    """
    This function plots the magnitude of the correction from the SDC method and JFNK method for \
    each iteration.

    :param data: errors from the simulations
    :param list titles: the titles of the subfigures
    :param list labels: the names of the lines
    :param bool do_legend: this flag indicates whether or not a legend will be shown
    :param str main_title: the title of the figure
    :param str xlabel: the x-axis label
    :param str ylabel: the y-axis label
    :param str ls: the line style

    :return: None
    """

    ndata = len(data)

    fig, axes = plt.subplots(ncols=ndata, sharey=True)

    fig.suptitle(main_title)
    lines = []

    if labels is None:
        labels = [None] * len(data[0].T)

    # get the data for each solution
    for ax, title, d in zip(axes, titles, data):

        ax.set_title(title)

        # plot the lines in each graph
        for y, label in zip(d.T, labels):
            temp = ax.plot(np.log10(y), ls, label=label)
            lines.append(temp[0])
            ax.set_xlabel(xlabel)

    # plot the legend
    if do_legend:
        fig.legend(lines, labels, loc='best')

    # set the y-axis label
    ax = axes[0]
    ax.set_ylabel(ylabel)

    return