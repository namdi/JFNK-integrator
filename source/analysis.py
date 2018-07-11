# This file was written by Dr. Namdi Brandon
# ORCID: 0000-0001-7050-1538
# June 29, 2018

"""
This module contains class :class:`analysis.Result` that help saves the information \
related to the ODE solution. Also, this module contains functions that aid in \
analyzing the results of the various solutions.
"""

# ===============================================
# import
# ===============================================
import numpy as np
import numpy.linalg as LA

import copy

# ===============================================
# class Result
# ===============================================
class Result(object):

    """
    This class saves the information related to the ODE solution and the parameters used in order \
    to generate the solution.

    :param params.Params p: the parameters related to the solver
    :param float t0: the initial time :math:`t_{init}`
    :param float t_final: the final time :math:`t_{final}`
    :param numpy.ndarray t: the time nodes used in the simulation
    :param numpy.ndarray y: the approximation at the time nodes
    :param bool is_adaptive: a flag indicating whether the simulation used adaptive (if True) \
    step sizes or uniform step sizes (if False)
    :param Y: a history of the solution  history for each iteration at each time step
    :param D: a history of the deferred correction for each iteration at each time step
    :param float h: the adaptive time steps used
    :param float dt_init: the initial step size :math:`\\Delta{t}_{init}` used in the adaptive scheme
    :param float be_tol: the convergence criteria for the backward Euler solver
    :param float sdc_tol: the convergence criteria for the SDC solver
    :param float tol: the approximated absolute error at each step for the adaptive solution
    """

    def __init__(self, p, t0, t_final, t, y, is_adaptive, Y=None, D=None, h=None, dt_init=None, be_tol=None,
                 sdc_tol=None, tol=None):


        # the CMAQ case

        # the parameters
        self.p = copy.copy(p)

        # the initial time entered to the solver
        self.t0 = t0

        # the final time entered to the solver
        self.t_final = t_final

        # the time steps
        self.t = t

        # the solution, dimensions (total number of nodes, size of problem)
        self.y = np.array(y)

        # indicate whether results come from an adaptive algorithm
        self.is_adaptive = is_adaptive

        # the history of the solution (i.e, for each time step, the solution for each iteration)
        if Y is not None:
            Y = copy.copy(Y)
        self.Y = Y

        # the history of the corrections(i.e, for each time step, the solution for each iteration)
        if D is not None:
            D = copy.copy(D)
        self.D = D

        # the time steps used
        if type(h) is np.ndarray:
            h = np.array(h)
        self.h = h

        # the initial step size if for adaptive
        self.dt_init = dt_init

        # backward euler solver tolerance
        self.be_tol = be_tol

        # sdc iteration tolerance
        self.sdc_tol = sdc_tol

        # the tolerance for adaptive time stepping
        self.tol = tol

        n_sdc_per_step, n_sdc = None, None

        if self.Y is not None:
            n_sdc_per_step = np.array([len(x) for x in Y])
            n_sdc = n_sdc_per_step.sum()

        self.n_sdc_per_step = n_sdc_per_step
        self.n_sdc = n_sdc

        return

# ===============================================
# functions
# ===============================================
def analyze_corrections(D, Y):

    """
    This function calculates the following

    #. for each iteration :math:`k`, the magnitude of the correction for each component \
    :math:`i`: :math:`\\| \\delta^{[k]}_i \\|`
    #. for each iteration :math:`k`, the magnitude of the correction for each component \
    :math:`i` on a :math:`log_{10}` scale : :math:`log_{10} \\left( \\| \\delta^{[k]}_i \\| \\right)`
    #. for each iteration :math:`k`, the magnitude of the relative correction for each \
    component :math:`i`: :math:`\\frac{\\| \\delta^{[k]}_i \\| }{ \\| y^{[k]}_i \\| }`
    #. for each iteration :math:`k`, the magnitude of the relative correction for each \
    component :math:`i` on a :math:`log_{10}` scale: \
    :math:`log_{10} \\left( \\frac{\\| \\delta^{[k]}_i \\| }{ \\| y^{[k]}_i \\| }\\right)`

    :param list D: corrections, dimensions (n iterations, p nodes, m problem size)
    :param list Y: approximations, dimensions (n iterations, p nodes, m problem size)

    :return: for each iteration the magnitude of the correction for each component, \
    for each iteration the magnitude of the correction for each component in log \
    base 10, \
    for each iteration the magnitude of the relative correction for each component, \
    for each iteration the magnitude of the relative correction for each component \
    in log base 10
    :rtype: numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray
    """

    y_norm      = np.vstack([np.linalg.norm(x, axis=0) for x in Y])
    d_norm      = np.vstack([np.linalg.norm(x, axis=0) for x in D])
    d_norm_rel  = np.divide(d_norm, y_norm)

    log_d_norm      = np.log10(d_norm)
    log_d_norm_rel  = np.log10(d_norm_rel)

    return d_norm, log_d_norm, d_norm_rel, log_d_norm_rel


def error_analysis(y_approx, y_solution, threshold=1e-20):

    """
    This function calculates the absolute error or the relative error.

    :param numpy.ndarray y_approx: the approximate solution
    :param numpy.ndarray y_solution: the more accurate solution
    :param float threshold: the threshold to set the components to 0

    :return: the absolute error, the relative error
    :rtype: numpy.ndarray, numpy.ndarray
    """

    aerr = np.abs(y_approx - y_solution)

    # the bottom (denominator)
    bot = np.abs(y_solution)

    rerr = aerr / bot

    # indices of zero entries
    idx = bot <= threshold
    rerr[idx] = 0

    return aerr, rerr

def get_correction_norms(Y, D):

    """
    This function calculates various measures of the corrections from SDC sweep

    #. for each iteration :math:`k`, the maximum magnitude of the relative correction for each \
    component :math:`i`: :math:`\\max \\frac{\\| \\delta^{[k]}_i \\| }{ \\| y^{[k]}_i \\| }`
    #. for each iteration :math:`k`, the maximum magnitude of the correction for each component \
    :math:`i`: :math:`\\max \\| \\delta^{[k]}_i \\|`
    #. for each iteration :math:`k`, the mean magnitude of the relative correction for each \
    component :math:`i`: :math:`E\\left[\\frac{\\| \\delta^{[k]}_i \\| }{ \\| y^{[k]}_i \\| }\\right]`
    #. for each iteration :math:`k`, the mean magnitude of the correction for each component \
    :math:`i`: :math:`E[\\| \\delta^{[k]}_i \\|]`

    :param list Y: the approximations for the solution at each iteration arrays of dimensions (n nodes, size of problem)
    :param list D: the corrections for each iteration arrays of dimensions (n nodes, size of problem)

    :return: the maximum relative norm, the maximum absolute norm, the mean relative norm, \
    the mean absolute norm
    :rtype: numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray
    """

    # calculate the relative correction for each component
    f = lambda y, d: np.linalg.norm(d, axis=0) / np.linalg.norm(y, axis=0)
    g = lambda y: np.linalg.norm(y, axis=0)

    # this avoid division be zero since we are getting data from relative errors. avoids values from  \
    # zero-valued solutions. Avoids nan's from relative errors that are infinity
    f_rel = lambda x: x[np.isfinite(x)]

    rel_max = np.array([f_rel(f(y, d)).max() for y, d in zip(Y, D)])
    ab_max = np.array([g(d).max() for d in D])

    rel_mean = np.array([f_rel(f(y, d)).mean() for y, d in zip(Y, D)])
    ab_mean = np.array([g(d).mean() for d in D])

    return rel_max, ab_max, rel_mean, ab_mean

def get_error_time_nodes(t, y, y_spline, do_relative=True):

    """
    This function calculates the absolute or relative error comparing the approximate \
    solution and the "exact" solution at all of the temporal nodes.

    :param numpy.ndarray t: the temporal nodes on the approximate solution
    :param numpy.ndarray y: the approximate solution
    :param function y_spline: a function that may interpolate the "exact" (more accurate) solution
    :param bool do_relative: a flag indicating whether (if True) to calculate the relative error \
    or not to (if False) to calculate the absolute error

    :return: the absolute or relative errors
    :rtype: numpy.ndarray
    """

    if do_relative:
        # get the raltive error for each time node
        err = [ relative_norm( y_spline(t[i]) - y[i], y_spline(t[i]) ) for i in range(len(t)) ]
    else:
        # get the absolute error for each time node
        err = [ LA.norm( y_spline(t[i]) - y[i], ord=np.inf ) for i in range(len(t)) ]

    err = np.array(err)

    return err

def get_error_time_steps(t, y, y_spline, n_nodes, do_relative=True):

    """
    This function calculates the absolute or relative error comparing the approximate \
    solution and the "exact" solution at the end of each time step.

    :param numpy.ndarray t: the temporal nodes
    :param numpy.ndarray y: the approximate solution
    :param function y_spline: a function that may interpolate the "exact" (more accurate) solution
    :param int n_nodes: the number of nodes per time step
    :param bool do_relative: a flag indicating whether (if True) to calculate the relative error \
    or not to (if False) to calculate the absolute error

    :return: the absolute or relative error
    :rtype: numpy.ndarray
    """
    err = list()

    n_steps = (len(t) - 1) / (n_nodes - 1)
    n_steps = int(n_steps)

    for j in range(n_steps):

        # get the value at the end of the time step
        i = (n_nodes - 1) * j

        if do_relative:
            # calculate the relative error
            x = relative_norm( y_spline(t[i]) - y[i], y_spline(t[i]) )
        else:
            # calculate the absolute error
            x = LA.norm( y_spline(t[i]) - y[i], ord=np.inf )

        err.append(x)

    err = np.array(err)

    return err


def relative_norm(top, bot):

    """
    This function calculates the relative ratios between norms of vectors. Given two sets of vectors

    .. math::
        top_{n \\times m}, bot_{n \\times m}

    where :math:`n` is the number of temporal nodes and :math:`m` is the size of the system.

    For each component :math:`j`, calculate the relative norms over the time nodes

    .. math::
        ratio_j = \\frac{ \\| top_j \\|_2 }{\\| bot_j \\|_2}

    Make sure, we avoid division by zero

    .. math::
        x_j =
        \\begin{cases}
            ratio_j & \\text{if } \\|bot_j \\|_2 \\neq 0 \\\\
            \\| top_j \\|_2 & \\text{if } \\| bot_j \\|_2 = 0
        \\end{cases}

    Return the maximum value

    .. math::
        \\| x \\|_{\\infty}

    :param top: the top vector for a given iteration dimensions (n nodes, m size of problem)
    :param bot: the bottom vector for a given iteration  dimensions (n nodes, m size of problem)

    :return: the maximum value of the relative norm between two vectors
    :rtype: float
    """

    # the norm of the correction vector
    top_norm = LA.norm(top, axis=0)

    # the norm of the approximation vector
    bot_norm = LA.norm(bot, axis=0)

    # the relative correction magnitude
    ratio = top_norm / bot_norm

    # indices of components that have a non-zero denominator
    idx = np.isfinite(ratio)

    if idx.all():
        # not dividing by zero
        value = ratio.max()
    else:
        # at least, one component is dividing by zero
        # use the absolute correction for those components diving by zero
        # use the relative correction for those components NOT dividing by zero
        value = max(top_norm[~idx].max(), ratio[idx].max())

    return value


def run_threshold(y, threshold):

    """
    If the value of the solution is below the threshold, set it to zero.

    :param numpy.ndarray y: the approximate solution
    :param float threshold: the threshold

    :return: None
    """
    y[np.abs(y) < threshold] = 0

    return