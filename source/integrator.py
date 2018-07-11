# This file was written by Dr. Namdi Brandon
# ORCID: 0000-0001-7050-1538
# June 29, 2018

"""
This module contains code for functions for solving temporal ordinary
differential equations (ODEs)

.. math::
    \\frac{dy(t)}{dt} &= f(t, y(t)) \\\\
    y(0) &= y_0

which has the following solution

.. math::
    y(t) = y_0 +  \int^t_0 \\, f(\\tau, y(\\tau) ) \\, \\mathrm{d}\\tau

The numerical integrators are the following:

* Backward (implicit) Euler method
* Jacobian-Free Newton-Krylov (JFNK) method with uniform time stepping
* Jacobian-Free Newton-Krylov (JFNK) method with adaptive time stepping
* Spectral Deferred Corrections (SDC) (implicit)
* Spectral solution (Gauss collocation formulation) solver

+---------------+-----------------------------------+
| Abbreviations | Meaning                           |
+===============+===================================+
| JFNK          | Jacobian-Free Newton Krylov       |
+---------------+-----------------------------------+
| ODE           | Ordinary differential equation    |
+---------------+-----------------------------------+
| SDC           | Spectral deferred corrections     |
+---------------+-----------------------------------+

.. moduleauthor:: Dr. Namdi Brandon
"""

# ===============================================
# import
# ===============================================
import numpy as np
import numpy.linalg as LA

from scipy.optimize import newton_krylov
from scipy.optimize import anderson
from scipy.optimize import root

import time

# this comes from the pyweno code from Matt Emmett
import points

# ===============================================
# constants
# ===============================================


# newton krylov function RELATIVE tolerance in residual
NK_FTOL = 1e-1

# backward euler function RELATIVE tolerance
BE_TOL = 1e-12

# the convergence criteria for RELATIVE strength of corrections for SDC
SDC_TOL = 1e-14

# the maximum amount of SDC iterations
N_ITER_MAX_SDC = 500

# the maximum anount of JFNK iterations
N_ITER_MAX_NEWTON = 50

# this is for debugging
N_STEPS_MAX_ADAPTIVE = int(1e9)

# the largest multiplier that a time step can grow by
ADAPTIVE_SCALER_MAX = 4

# the smallest multiplier that a time step can grow by
ADAPTIVE_SCALER_MIN = 1.5
# ===============================================
# functions
# ===============================================

def adjust_scaler(x, x_min=ADAPTIVE_SCALER_MIN, x_max=ADAPTIVE_SCALER_MAX):

    """
    For the adaptive time stepping algorithm, this function adjusts the adaptive step size

    .. math::
        x \leftarrow
        \\begin{cases}
            x & \\text{if } x < 1 \\\\
            1 & \\text{if } 1 \\le x \\le x_{min} \\\\
            \\min(x, x_{max}) & \\text{if } x > x_{min}
        \\end{cases}

    :param float x: the ratio in which to grow or shrink the time step
    :param float x_min: the minimum ratio in which to grow a time step
    :param float x_max: the maximum ratio in which to grow a time step

    :return: the ratio in which to grow or shrink the time step
    :rtype: float
    """

    if x > x_min:
        # upper bound. the maximum scaling multiplier is scaler_max. This avoids taking too large a step size
        x = min(x, x_max)

    elif (x >= 1) and (x <= x_min):
        x = 1.0

    return x

def backward_euler(f_eval, t, y0, S, be_tol=BE_TOL, do_print=False):

    """
    This function runs the backward Euler method over all of the nodes :math:`t_i` in an entire time step
    of size :math:`\\Delta{t}`.

     .. math::
        y - \\Delta{t}\\tilde{S}F(y) = rhs

    :param function f_eval: the derivative function :math:`y' = f(t,y)`.
    :param numpy.ndarray t: the time nodes over a time step :math:`\\Delta{t}`
    :param numpy.ndarray y0: the initial condition ( length = m)
    :param numpy.ndarray S: the backward Euler integration matrix :math:`\\tilde{S}`
    :return:
    """

    # start timing
    start = time.time()

    # the number of nodes
    n_nodes = len(t)

    # the size of the system
    m = len(y0)

    # the solution over the time step
    y = np.zeros((n_nodes, m))

    #
    # run backward euler over a full time step
    #
    for i in range(n_nodes):

        h = S[i, i]

        # if using left end point (e.g. gauss-lobatto nodes)
        if i == 0 and h == 0:
            y[0, :] = y0[:]
        else:
            # if not using left end point
            if i == 0 and h != 0:
                rhs = y0[:]
            else:  # all other points
                rhs = y[i - 1, :]

            # solve backward euler for the ith node
            y[i, :] = backward_euler_node(f_eval, t=t[i], h=h, rhs=rhs, x0=rhs, be_tol=be_tol)

    # end timing
    end = time.time()

    if do_print:
        print_elapsed_time(start, end)

    return y

def backward_euler_node(f_eval, t, h, rhs, x0, be_tol=BE_TOL):

    """
    This function solves an ODE system using backward Euler method at a specific time :math:`t`. This code uses
    a general numerical solver in order to do the inversion in the backward Euler method in order
    to find :math:`y_i`.

    .. math::
        y_i - h_i f(t_i, y_{i-1}) = rhs_i

    :param function f_eval: the derivative function :math:`y' = f(t,y)`.
    :param float t: the time at a node
    :param float h: the step size
    :param numpy.ndarray rhs: the right hand side of the backward euler system [m x 1]
    :param numpy.ndarray x0: the initial guess for the solution [m x 1]
    :param float be_tol: the tolerance for the backward Euler solver

    :return: the approximate solution [m x 1]
    :rtype: numpy.ndarray
    """

    # the approximate solution
    y       = np.zeros( rhs.shape )

    if h == 0:
        y[:] = rhs[:]
    else:

        # find the root of this system
        A = lambda x: x - h*f_eval(t, x) - rhs

        #y[:] = newton_krylov(A, x0, f_rtol=be_tol)
        y[:] = root(A, x0, tol=be_tol).x

    return y



def convergence_criteria(d, y, tol):

    """
    This function calculates whether or not the correction vector :math:`\\delta` is small enough to
    satisfy the convergence criteria.

    .. math::
        x = \\frac{ \\displaystyle{ \\| \\delta \\| } }{ \\displaystyle{ \\| y \\| } } \\\\
        \\begin{cases}
            x \\le tol & \\text{converged} \\\\
            x > tol & \\text{not converged} \\\\
        \\end{cases}

    :param numpy.ndarray d: the correction vector :math:`\\delta` for a given iteration dimensions (n nodes, size of problem)
    :param numpy.ndarray y: the approximate solution :math:`y` for a given iteration  dimensions (n nodes, size of problem)
    :param float tol: the correction tolerance for the convergence criteria

    :return: a flag indicating whether or not the corrections are small enough to qualify for convegence
    :rtype: bool
    """

    value = relative_norm(d, y)

    # the method is converged if the tolerance is low
    is_converged = value <= tol

    return is_converged

def jfnk(f_eval, t, y0, y_approx, S, S_p, spectral_radius, n_iter_max_newton=N_ITER_MAX_NEWTON, \
             be_tol=BE_TOL, sdc_tol=SDC_TOL, do_print=False):

    """
    The Jacobian-Free Newton-Krylov (JFNK) method to approximate a solution to the spectral solution

     .. math::
        y - \\Delta{t}SF(y) = y_0

    over one time step of size :math:`\\Delta{t}`. This is done by use a modified version of Newton's method to
    find a calculate a solution

    .. math::
        H(y) = 0

    where :math:`H(y^{[k]}) = \\delta^{[k]}` corresponds to one iteration of the SDC method.

    Given :math:`y^{[0]}`, this method does the following

    1. calculate the initial SDC iterations

        .. math::

            \\begin{cases}
	            \\delta^{[k]} &= H(y^{[k]})  \\text{\indent calculate an SDC correction} \\\\
	            y^{[k+1]} &= y^{[k]} + \\delta^{[k]} \\text{\\indent update the SDC solution}
            \\end{cases}

        until the the solution converges or order convergence has been observed

    2. do the Newton (Jacobian-Free) iterations

        .. math::

            J_{H}(y^{[p]})\\Delta{x} &= -H(y^{[p]})   \\\\
            \\implies J_{H}(y^{[p]})\\Delta{x} &= -\\delta^{[p]}   \\\\

        Set :math:`\\Delta{x} = \\sum^{p-1}_{j=0} c_j \\delta^{[j]}` and solve

        .. math::
            J_{H}(y^{[p]})\\sum^{p-1}_{j=0} c_j \\delta^{[j]} &= -\\delta^{[p]}  \\\\
            \\implies \\sum^{p-1}_{j=0} c_j (\\delta^{[k+1]} - \\delta^{[k]}) &= -\\delta^{[p]} \\\\

        Solve the system for the Jacobian-Free system
            .. math::
                \\begin{cases}
                    Ac &= -\\delta^{[p]} \\\\
                    y &\\leftarrow y^{[p]} + \\sum^{p-1}_{j=0} c_j \\delta^{[j]}
                \\end{cases}

    :param function f_eval: the derivative function :math:`y' = f(t,y)`.
    :param numpy.ndarray t: the time nodes over a time step :math:`\\Delta{t}`
    :param numpy.ndarray y0: the initial condition ( length = m)
    :param numpy.ndarray y_approx: the provisional solution (dimensions n time nodes, size of the problem)
    :param numpy.ndarray S: the spectral integration (Gaussian quadrature) matrix, :math:`S`
    :param numpy.ndarray S_p: the backward Euler integration matrix, :math:`\\tilde{S}`
    :param float spectral_radius: the spectral radius of the correction matrix for the extremely stiff case
    :param int n_iter_max_newton: the maximum number of Newton iterations
    :param float be_tol: the convergence criteria for the backward Euler solver
    :param float sdc_tol: the convergence criteria for the SDC solver
    :param bool do_print: a flag indicating whether or not to print the elapsed time

    :return: the solution, the history of approximations for each iteration, the history of corrections \
    for each iteration

    :rtype: numpy.ndarray (dimensions n time nodes, size of the problem) , list (length number of iterations), \
    list (length of iterations), bool, bool, numpy.ndarray (length of iterations)
    """

    # start timing
    start = time.time()

    # initial SDC iterations before order convergence (due to stiffness)
    y_init, Y_init, D_init, is_converged, is_stiff, ratios \
        = jfnk_initial(f_eval, t, y0, y_approx, S, S_p, spectral_radius, be_tol=be_tol, sdc_tol=sdc_tol)

    Y, D = list(), list()

    # store the solution
    if is_converged:
        y = y_init

    # once order convergence is detected use JFNK
    if (not is_converged) and (is_stiff):
        y, Y, D, is_converged = jfnk_iterations(f_eval, t, y0, y_init, S, S_p, n_iter_max_newton, \
                                     be_tol=be_tol, sdc_tol=sdc_tol)

    # store the history of the solutions
    Y = [Y_init, Y]
    Y = [subitem for item in Y for subitem in item]

    # store the history of corrections
    D = [D_init, D]
    D = [ subitem for item in D for subitem in item]

    # stop timing
    end = time.time()

    if do_print:
        print_elapsed_time(start, end)

    return y, Y, D, is_converged, is_stiff, ratios

def jfnk_adaptive(f_eval, t_init, t_final, dt_init, p, y0, tol, n_iter_max_newton=N_ITER_MAX_NEWTON, be_tol=BE_TOL, \
                  sdc_tol=SDC_TOL, n_steps_max=N_STEPS_MAX_ADAPTIVE, do_print=False,):
    """
    Run the JFNK with adaptive step sizes from :math:`t \\in [t_{init}, t_{final}]` to calculate an
    approximation to the solution

    .. math::
        y(t_{final}) = y(t_{init}) +  \int^{t_{final}}_{t_{init}} \\, f(\\tau, y(\\tau) ) \\, \\mathrm{d}\\tau

    Such that for each time step the step size :math:`\\Delta{t}` is chosen so that the difference between
    the exact solution :math:`y` and the approximate solution :math:`\\tilde{y}`

    .. math::
        \||y - \\tilde{y} \\|_{\\infty} \\le tol

    :param function f_eval: the derivative function :math:`y' = f(t,y)`.
    :param float t_init: the initial time :math:`t_{init}`
    :param float t_final: the final time :math:`t_{final}`
    :param float dt_init: the initial step size :math:`\\Delta{t}_{init}`
    :param params.Params p: the parameters related to the solver
    :param numpy.ndarray y0: the initial condition :math:`y(t_{init})` (length = m)
    :param float tol: the approximated absolute error at each step for the adaptive solution
    :param int n_iter_max_newton: the maximum number of Newton iterations
    :param float be_tol: the convergence criteria for the backward Euler solver
    :param float sdc_tol: the convergence criteria for the SDC solver
    :param int n_steps_max: the maximum number of steps in the solver
    :param bool do_print: a flag indicating whether or not to print the elapsed time

    :return: all of the time nodes, the value of the solution at each node, a history of the solution \
    history for each iteration at each time step, history of the deferred correction for each \
    iteration at each time step, the step size for each time step

    :rtype: numpy.ndarray, numpy.ndarray, list, list, numpy.ndarray
    """

    assert t_final > t_init

    assert dt_init <= (t_final - t_init)

    # start timing
    start = time.time()

    # integration step size
    h = dt_init

    # the number of temporal nodes per time step
    n_nodes = p.n_nodes

    # the type of nodes
    node_type = p.node_type

    # a listing
    # D_all: list length of n_steps
    # containing a list length of iterations
    # each iteration contains the the correction vectors (n_nodes, size of problem)
    y_all, t_all, Y_all, D_all = list(), list(), list(), list()

    y_all.append(y0)
    t_all.append(np.array(t_init))
    h_all = list()

    # the steps counter
    j = 0

    # the initial time for the time step
    t0 = t_init

    # the initial condition
    v0 = np.array(y0)

    do_stop = False

    scaler_big_enough = 1.5

    while (t0 < t_final) and (not do_stop):

        assert j < n_steps_max, 'Done too many time steps: ' + str(j) + '. Quiting!'

        # run the jfnk with 1 step
        t1, y1, Y1, D1 = jfnk_uniform(f_eval, t_init=t0, t_final=(t0 + h), n_steps=1, p=p, y0=v0, \
                                      n_iter_max_newton=n_iter_max_newton, be_tol=be_tol, sdc_tol=sdc_tol)

        # run the jfnk with 2 steps
        t2, y2, Y2, D2 = jfnk_uniform(f_eval, t_init=t0, t_final=(t0 + h), n_steps=2, p=p, y0=v0, \
                                      n_iter_max_newton=n_iter_max_newton, be_tol=be_tol, sdc_tol=sdc_tol)

        # run the jfnk with h/2
        scaler = step_size_scaler(y1[-1], y2[-1], n_nodes, 2, tol, node_type)

        # adjust the scaler
        scaler = adjust_scaler(scaler, scaler_big_enough)

        # only rerun integration if it causes a small step size OR scaler is large enough
        do_rerun = (scaler < 1) or (scaler > scaler_big_enough)

        # rerun the integration using the new step size
        if do_rerun:

            # use the new step size
            h = update_step_size(scaler * h, t0, t_final)

            # rerun the JFNK with the improved time step
            t, y, Y, D = jfnk_uniform(f_eval, t_init=t0, t_final=(t0 + h), n_steps=1, p=p, y0=v0, \
                                      n_iter_max_newton=n_iter_max_newton, be_tol=be_tol, sdc_tol=sdc_tol)
        else:
            # use the original approximation
            t, y, Y, D = t1, y1, Y1, D1

        # store the values from this time step
        if p.t[-1] == 1:
            y_all.append(y[1:])
            t_all.append(t[1:])
        else:
            y_all.append(y)
            t_all.append(t)

        h_all.append(h)
        Y_all.append(Y[0])
        D_all.append(D[0])

        #
        # update the initial time for the next time step
        #
        t0 = t0 + h

        # update the initial condition for the next time step
        v0 = np.array(y[-1])

        # this means that we have arrived at the end
        #do_stop = t0 >= t_final
        do_stop = stopping_criteria(t_final, t0)

        # this prevents over shooting t_final,
        h = update_step_size(h, t0, t_final)

        # update the counter
        j = j + 1

    # store the solution and time
    y_all = np.vstack(y_all)
    t_all = np.hstack(t_all)
    h_all = np.hstack(h_all)

    # end timing
    end = time.time()

    if do_print:
        print_elapsed_time(start, end)

    return t_all, y_all, Y_all, D_all, h_all


def jfnk_initial(f_eval, t, y0, y_approx, S, S_p, spectral_radius, be_tol=BE_TOL, sdc_tol=SDC_TOL, \
                 n_iter_max=N_ITER_MAX_SDC, do_print=False):

    """
    This function runs initial SDC iterations until order convergence is observed. Given
    :math:`H(y^{[k]}) = \\delta^{[k]}` corresponds to one iteration of the SDC method.

    1. Run 2 initial SDC iterations

    .. math::

		\\begin{cases}
		    \\delta^{[k]} & \\leftarrow H(y^{[k]}) \\\\
		    y^{[k+1]} & \\leftarrow y^{[k]} + \\delta^{[k]},  k=0, 1
		\\end{cases}

    2. Caculate the ratio of the corrections

    .. math::
        r = \\frac{\\| \\delta^{[k-1]} \\|_F}{ \\| \\delta^{[k-2]} \\|_F}

    where :math:`\\| \\cdot \\|_F` is the Frobenius norm.

    If :math:`\\frac{r}{ \\rho(C_s) } > 0.1`, order convergence is observed. Stop the function.

    If :math:`\\frac{r}{ \\rho(C_s) } \\le 0.1`, if not converged, do another SDC iteration and go to step 2.

    :param function f_eval: the derivative function :math:`y' = f(t,y)`.
    :param numpy.ndarray t: the time nodes over a time step :math:`\\Delta{t}`
    :param numpy.ndarray y0: the initial condition
    :param numpy.ndarray y_approx: the provisional solution (dimensions n time nodes, size of the problem)
    :param S: the spectral (Gaussian quadrature) integration matrix
    :param S_p: the backward Euler integration matrix
    :param float spectral_radius: the spectral radius of the correction matrix for the extremely stiff case
    :param float be_tol: the convergence criteria for the backward Euler solver
    :param float sdc_tol: the convergence criteria for the SDC solver
    :param n_iter_max: the maximum number of SDC iterations
    :param bool do_print: a flag indicating whether or not to print the elapsed time

    :return: the approximation, the history of the approximations for each iteration, the history of the \
    deferred corrections for each iteration, a flag indicating whether or not the solution has converged, \
    a flag indicating whether or not the problem is stiff, relative magnitude of consecutive iterations

    :rtype: numpy.ndarray, list, list, bool, bool, numpy.ndarray
    """

    # flag for dealing with a stiff system
    is_stiff = False

    # counter
    i = 0

    # a list of the ratios of d(k+1) / d(k)
    ratios = []

    # run an initial 2 iterations
    y_sdc, Y_sdc, D_sdc, is_converged = sdc(f_eval, t, y0, y_approx, S, S_p, n_iter_max_sdc=2, be_tol=be_tol, \
                                            sdc_tol=sdc_tol, do_print=False)

    # store the previous approximations
    Y = [x for x in Y_sdc]

    # store the previous corrections
    D = [x for x in D_sdc]

    if is_converged:
        ratios = np.array( [] )

    # JFNK initial solutions
    while (not is_converged) and (not is_stiff) and (i < n_iter_max):

        # create a way to handle a component that has converged
        # d(k+1) / d(k)

        # antiquated
        # ratio = relative_norm(D[-1], D[-2])

        # use Frobenius matrix norm over all correction vectors
        ratio = LA.norm(D[-1], ord='fro') / LA.norm(D[-2], ord='fro')

        ratios.append(ratio)

        is_stiff = (ratio / spectral_radius) > 0.1

        if is_stiff:
            if do_print:
                print('Stiff (order convergence detected). Use JFNK.')
        else:
            if do_print:
                print('Non-stiff. Use SDC')

            y_sdc, Y_sdc, D_sdc, is_converged = sdc(f_eval, t, y0, y_sdc, S, S_p, n_iter_max_sdc=1, \
                                                    be_tol=be_tol, sdc_tol=sdc_tol, do_print=False)
            # store the approximation
            Y.append(Y_sdc[0])

            # store the
            D.append(D_sdc[0])

        # update counter
        i = i + 1

    if len(ratios) != 0:
        ratios = np.vstack(ratios)

    return y_sdc, Y, D, is_converged, is_stiff, ratios

def jfnk_iterations(f_eval, t, y0, y_init, S, S_p, n_iter_max_newton=N_ITER_MAX_NEWTON, be_tol=BE_TOL, sdc_tol=SDC_TOL):

    """
    This function solves the Newton's method iterations for solving

    .. math::
        H(y) = 0

    where :math:`H(y^{[k]}) = \\delta^{[k]}` corresponds to one iteration of the SDC method.

    1. Run :math:`n+1` SDC iterations.

    .. math::

        \\begin{cases}
		    \\delta^{[k]} & \\leftarrow H(y^{[k]}) \\\\
		    y^{[k+1]} & \\leftarrow y^{[k]} + \\delta^{[k]},  k=0, \\ldots, n
		\\end{cases}

    2. Solve the Newton iteration system without using the Jacobian explicitly

    .. math::

            J_{H}(y^{[n]})\\Delta{x} &= -H(y^{[n]})   \\\\
            \\implies J_{H}(y^{[n]})\\Delta{x} &= -\\delta^{[n]}   \\\\

    Set :math:`\\Delta{x} = \\sum^{n-1}_{j=0} c_j \\delta^{[j]}` and solve

    .. math::
        J_{H}(y^{[n]})\\sum^{n-1}_{j=0} c_j \\delta^{[j]} &= -\\delta^{[n]}  \\\\
        \\implies \\sum^{n-1}_{j=0} c_j (\\delta^{[k+1]} - \\delta^{[k]}) &= -\\delta^{[n]} \\\\

    Solve the system for the Jacobian-Free system

    .. math::
        \\begin{cases}
            Ac &= -\\delta^{[n]} \\\\
            y &\\leftarrow y^{[n]} + \\sum^{n-1}_{j=0} c_j \\delta^{[j]}
        \\end{cases}

    3. If not converged, repeat by going to step 1.

    :param function f_eval: the derivative function :math:`y' = f(t,y)`.
    :param numpy.ndarray t: the time nodes in the time step (length n_nodes)
    :param numpy.ndarray y0: the initial solution [size of the problem x 1]
    :param numpy.ndarray y_init: the approximate solution [n_nodes x size of the problem]
    :param numpy.ndarray S: the spectral integration matrix 
    :param numpy.ndarray S_p: the preconditioner integration matrix 
    :param int n_iter_max_newton: the maximum number of Newton iterations
    
    :return: the solution, the history of the solutions for each iteration, the deferred \
    correction for each iteration, a flag indicating whether the procedure converged

    :rtype: numpy.ndarray, list, list, bool
    """

    # the current solution
    y = np.array( y_init )

    # the number of sdc iterations
    k_sdc = len(t) + 1

    # the size of the system
    m = len(y0)

    # history of the corrections and approximations
    D_list, Y_list = list(), list()

    k = 0
    is_converged = False

    # the newton iterations
    while (not is_converged) and (k < n_iter_max_newton):

        # do sdc iterations
        # D is [k_sdc x n_nodes x m]
        y, Y, D, is_converged = sdc(f_eval, t, y0, y, S, S_p, n_iter_max_sdc=k_sdc,\
                                    be_tol=be_tol, sdc_tol=sdc_tol)

        if is_converged:
            # store the solution history from the SDC sweeps as a list
            Y_list.append(Y)
            # store the history of the corrections
            D_list.append(D)
        else:
            # store the solution history from the SDC sweeps as a list
            Y_list.append(Y[:-1])
            # store the history of the corrections
            D_list.append(D[:-1])

            # this solution is used in the Krylov iteration
            y = Y[-1]

            #
            # set up
            #

            # transpose data
            y = y.T
            D = [x.T for x in D]

            # for the system to solve
            A = [ (D[j + 1] - D[j]) for j in range(k_sdc - 1)]

            # the "basis" of deferred correction vectors
            M = [ D[j] for j in range(k_sdc - 1) ]

            #
            # newton's method
            #

            # for each unknown
            for i in range(m):

                # solve Newton iteration
                B = [a[i] for a in A]
                B = np.vstack(B).T
                rhs = -D[-1][i]
                rhs = rhs.reshape( (len(rhs), 1) )

                # solve the system Bc = rhs
                c, res, rank, s = np.linalg.lstsq(B, rhs)

                # create update dy = Vc
                V = [x[i] for x in M]
                V = np.vstack(V).T
                dy = V.dot(c)

                # update
                y[i, :] += dy[:].flatten()

            # logistics
            y = y.T

        # update
        k = k + 1

    # store the full history in one list
    Y_list = [item for subitem in Y_list for item in subitem ]

    D_list = [item for subitem in D_list for item in subitem]

    return y, Y_list, D_list, is_converged

def jfnk_step(f_eval, p, dt, t, y0,  n_iter_max_newton=N_ITER_MAX_NEWTON, be_tol=BE_TOL, \
              sdc_tol=SDC_TOL, do_print=False):
    """
    This function runs everything needed for the JFNK to run for 1 time step (i.e., \
    the backward euler precondition and the JFNK iterations).

    :param function f_eval: the derivative function :math:`y' = f(t,y)`.
    :param params.Params p: the parameters related to the solver
    :param float dt: the time step :math:`\\Delta{t}`
    :param numpy.ndarray t: the number of temporal nodes (length number of time nodes)
    :param numpy.ndarray y0: the initial condition (length, size of the problem)
    :param n_iter_max_newton: the maximum number of Newton iteration
    :param float be_tol: the convergence criteria for the backward Euler solver
    :param float sdc_tol: the convergence criteria for the SDC solver
    :param bool do_print: a flag indicating whether or not to print the elapsed time

    :return: the approximation, the history of the approximations for each iteration, the history of the \
    deferred corrections for each iteration, a flag indicating whether or not the solution has converged, \
    a flag indicating whether or not the problem is stiff, relative magnitude of consecutive iterations

    :rtype: numpy.ndarray, list, list, bool, bool, numpy.ndarray
    """

    # start timing
    start = time.time()

    # the spectral and preconditoner integration matrices
    S, S_p = dt * p.S, dt * p.S_p

    # spectral radius
    spectral_radius = p.spectral_radius

    # backward euler approximation
    y_be = backward_euler(f_eval, t, y0, S_p, be_tol=be_tol)

    # JFNK solver
    y, Y, D, is_converged, is_stiff, ratios = jfnk(f_eval, t, y0, y_be, S, S_p, spectral_radius, \
                                                   n_iter_max_newton, be_tol=be_tol, sdc_tol=sdc_tol)

    # end timing
    end = time.time()

    if do_print:
        print_elapsed_time(start, end)

    return y, Y, D, is_converged, is_stiff, ratios

def jfnk_uniform(f_eval, t_init, t_final, n_steps, p, y0, n_iter_max_newton=N_ITER_MAX_NEWTON, \
                 be_tol=BE_TOL, sdc_tol=SDC_TOL, do_print=False):

    """
    Run the JFNK with uniform step sizes from :math:`t \\in [t_{init}, t_{final}]` to calculate an
    approximation to the solution

    .. math::
        y(t_{final}) = y(t_{init}) +  \int^{t_{final}}_{t_{init}} \\, f(\\tau, y(\\tau) ) \\, \\mathrm{d}\\tau

    :param function f_eval: the derivative function :math:`y' = f(t,y)`.
    :param float t_init: the initial time :math:`t_{init}`
    :param float t_final: the final time :math:`t_{final}`
    :param int nsteps: the number of steps
    :param params.Params p: the parameters related to the solver
    :param numpy.ndarray y0: the initial condition :math:`y(t_{init})` (length = m)
    :param int n_iter_max_newton: the maximum number of Newton iterations
    :param float be_tol: the convergence criteria for the backward Euler solver
    :param float sdc_tol: the convergence criteria for the SDC solver
    :param bool do_print: a flag indicating whether or not to print the elapsed time

    :return: all of the time nodes, the value of the solution at each node, a history of the solution \
    history for each iteration at each time step, history of the deferred correction for each \
    iteration at each time step

    :rtype: numpy.ndarray, numpy.ndarray, list, list
    """

    assert t_final > t_init

    # start time
    start = time.time()

    # integration step size
    dt = (t_final - t_init) / n_steps

    # initial condition
    v0 = np.array(y0)

    # the normalized nodes within range [0, 1]
    if p.t[-1] != 1:
        t_step = np.hstack( [p.t, 1.0] )
    else:
        t_step = np.array(p.t)

    # scale the nodes to be in the correct time step
    nodes = t_init + dt * t_step

    # list all the approximations and time values
    y_all, t_all, Y_all, D_all = list(), list(), list(), list()

    # store the initial approximation and time
    y_all.append(y0)
    t_all.append( np.array(t_init) )

    # loop through the time steps
    for k in range(n_steps):

        # JFNK solver using 1 step
        y, Y, D, is_converged, is_stiff, ratios = jfnk_step(f_eval, p, dt, nodes, v0, n_iter_max_newton, \
                                                            be_tol=be_tol, sdc_tol=sdc_tol)

        assert is_converged, 'the JFNK did not converge on time step: ' + str(k) + '. Quitting...'

        # store the values
        y_all.append(y[1:])
        t_all.append(nodes[1:])

        Y_all.append(Y)
        D_all.append(D)

        # update the initial values for time and the approximation
        v0 = np.array(y[-1])

        # update the time step
        nodes = nodes + dt

    # store the values
    y_all = np.vstack(y_all)
    t_all = np.hstack(t_all)

    # end the timing
    end = time.time()

    # print elapsed time
    if do_print:
        print_elapsed_time(start, end)

    return t_all, y_all, Y_all, D_all

def print_elapsed_time(start, end):

    """
    This function prints the elapsed time [s] between to time points.

    :param float start: the start time [s]
    :param float end: the end time [s]
    :return:
    """

    print('elapsed time: %.2f[s]' % (end - start))

    return

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

def run_spectral(f_eval, t, y0, S, tol, y_approx, do_print=False, verbose=False):

    start = time.time()

    y_spect = spectral(f_eval, t, y0, S, tol, y_approx=y_approx, verbose=verbose)

    end = time.time()

    if do_print:
        print_elapsed_time(start, end)

    return y_spect

def scaler_lobatto(aerr, n_nodes, k, tol):

    """
    This function calculates :math:`x` the amount the step size should increase \
    or decrease for the proper adaptive time step size for Gauss-Lobatto nodes.

    :param numpy.ndarray aerr: the maximum absolute error between the approximate solution :math:`y_h` \
    and the higher accuracy solution :math:`y_{h/k}`
    :param int n_nodes: the number of temporal nodes within the time step
    :param int k: the amount of mini time steps leading up to :math:`y(\\Delta{t})`
    :param float tol: the approximated absolute error at each step for the adaptive solution

    .. math::
       x =  \\left( tol \\frac{ 1 - (\\frac{1}{k})^{p} }{ \\| y_{h} - y_{h/k}\\|_{\\infty}} \\right)^{1/p}

    where :math:`p = 2 * n - 1` and :math:`n` is the number of Gauss-Lobatto nodes

    :return: the amount the current step size should increase or decrease for the proper adaptive time step
    :rtype: float
    """

    p = 2 * n_nodes - 1

    x = tol * (1 - (1 / k)** p) / aerr

    scaler = x ** (1 / p)

    return scaler

def sdc(f_eval, t, y0, y_old, S, S_p, n_iter_max_sdc=N_ITER_MAX_SDC, be_tol=BE_TOL, \
        sdc_tol=SDC_TOL, do_print=False):

    """
    This function runs the SDC method until convergence.

    .. math::
        \\begin{cases}
		    \\delta^{[k]} & \\leftarrow H(y^{[k]}) \\\\
		    y^{[k+1]} & \\leftarrow y^{[k]} + \\delta^{[k]}
		\\end{cases}

    where :math:`H(y^{[k]}) = \\delta^{[k]}` corresponds to one iteration of the SDC method.

    :param function f_eval: the derivative function :math:`y' = f(t,y)`.
    :param numpy.ndarray t: the temporal nodes (length number of nodes) over the time step of size :math:`\\Delta{t}`
    :param numpy.ndarray y0: the initial condition :math:`y(t_{init})` (length = size of the problem)
    :param numpy.ndarray y_old: the provisional solution (dimensions, size of the problem)
    :param numpy.ndarray S: the spectral integration (Gaussian quadrature) matrix, :math:`S`
    :param numpy.ndarray S_p: the backward Euler integration matrix, :math:`\\tilde{S}`
    :param int n_iter_max_sdc: the maximum number of SDC iterations
    :param float be_tol: the convergence criteria for the backward Euler solver
    :param float sdc_tol: the convergence criteria for the SDC solver
    :param bool do_print: a flag indicating whether or not to print the elapsed time

    :return: the SDC solution, a history of the approximate solution for each iteration, the history of \
    the deferred correction for each itearation, a flag indicating whether or not the solution has met the \
    convergence criteria

    :rtype: numpy.ndarray, list, list, bool
    """

    # start timing
    start = time.time()

    # the vector of derivatives at each time step
    F = np.zeros( y_old.shape )

    # the number of time nodes, the size of the system
    n_nodes, m = F.shape

    # the full history of solutions
    Y = list()

    # the correction history
    D = list()

    # the initial guess
    y_sdc = np.array(y_old)

    # the counter
    k = 0

    # a flag indicating whether or not SDC has converged
    is_converged = False

    while (not is_converged) and  (k < n_iter_max_sdc):

        # store the solution
        Y.append(y_sdc)

        F[:] = 0
        for i in range(n_nodes):
            F[i] = f_eval(t[i], y_sdc[i])

        y_sdc, d = sdc_sweep(f_eval, t, y0, y_sdc, F, S, S_p, be_tol=be_tol)

        # check to see if sdc has converged
        is_converged = convergence_criteria(d, Y[-1], tol=sdc_tol) # original
        #is_converged = convergence_criteria(d, y_sdc, tol=sdc_tol)

        # store the correction
        D.append(d)

        # update counter
        k = k + 1

    Y = np.vstack(Y)
    D = np.vstack(D)

    # reshape the arrays
    Y.resize( (k, n_nodes, m) )
    D.resize( (k, n_nodes, m) )

    # stop timing
    end = time.time()

    # print the elapsed time
    if do_print:
        print_elapsed_time(start, end)

    return y_sdc, Y, D, is_converged

def sdc_node(f_eval, t, h, rhs, y_old, be_tol=BE_TOL):

    """
    This function runs SDC on the :math:`i^{th}` time node within the time step.

     .. math::
            y^{[k+1]}_i - hF(y^{[k+1]}_i) = rhs_i

    where

    .. math::
        rhs_i = y^{[k]}_{i-1} + (\\Delta{t}S_i - he_i)\\cdot F(y^{[k]})

    :param function f_eval: the derivative function :math:`y' = f(t,y)`.
    :param float t: the time
    :param float h: the time step size
    :param numpy.ndarray rhs: the right hand side of the backward euler system [m x 1]
    :param numpy.ndarray y_old: the approximate solution before the update [number of nodes, size of the problem)
    :param float be_tol: the convergence criteria for the backward Euler solver

    :return: both the improved solutionand the correction at the time :math:`t`
    :rtype: numpy.ndarray, numpy.ndarray
    """

    # the improved solution at the time node t
    y_new = backward_euler_node(f_eval, t, h, rhs=rhs, x0=y_old, be_tol=be_tol)

    # deferred correction
    d = y_new - y_old

    return y_new, d



def sdc_sweep(f_eval, t, y0, y_old, F, S, S_p, be_tol=BE_TOL):

    """
    This runs SDC on the entire time step interval

    .. math::
        \\begin{cases}
            y^{[k+1]} - \\Delta{t}\\tilde{S}F(y^{[k+1]}) & = y_0 + \\Delta{t}(S- \\tilde{S})F(y^{[k]}) \\\\
            \\delta^{[k]} &=  H(y^{[k]}) = y^{[k+1]} - y^{[k]}
        \\end{cases}

    :param function f_eval: the derivative function :math:`y' = f(t,y)`.
    :param numpy.ndarray t: the time nodes over a time step :math:`\\Delta{t}`
    :param numpy.ndarray y0: the initial solutions length, size of the problem
    :param numpy.ndarray y_old: the approximate solution :math:`y^{[k]}` [n_nodes x m]
    :param numpy.ndarray F: the derivative at the approximate solution :math:`F(y^{[k]})`
    :param numpy.ndarray S: the spectral integration (Gaussian quadrature) matrix :math:`S`
    :param numpy.ndarray S_p: the backward Euler integration matrix :math:`\\tilde{S}`
    :param float be_tol: the tolerance for the backward Euler solver

    :return: the approximate solution, the correction
    :rtype: numpy.ndarray (dimensions number of temporal nodes x size of the problem), \
    numpy.ndarray (dimensions number of temporal nodes by size of the problem)
    """

    # the number of nodes
    n_nodes = len(t)

    # the size of the system
    m = len(y0)

    # the solution over the time step
    y = np.zeros( (n_nodes, m) )

    # the corrections over the time step
    d = np.zeros(y.shape)

    #
    # run sdc over a full time step
    #
    for i in range(n_nodes):

        # the step size in the preconditioner integration
        h = S_p[i, i]
        w = np.zeros(n_nodes)
        w[i] = h

        # skip if the temporal node includes the left end point of the time step
        do_skip =  (S_p[i] == 0).all()

        if i == 0 and do_skip:
            y[0,:] = y0[:]
        else:
            if (i == 0) and (not do_skip):
                rhs     = y0 + (S[i,:] -  w).dot(F)
            else:
                rhs     =  y[i-1] + (S[i,:] - S[i-1,:] - w).dot(F)

            # solve sdc at the time
            y_temp, d_temp = sdc_node(f_eval, t[i], h, rhs, y_old[i], be_tol=be_tol)

            # store the values
            y[i,:]  = y_temp
            d[i,:]  = d_temp

    return y, d

def spectral(f_eval, t, y0, S, f_tol=NK_FTOL, y_approx=None, do_print=False, verbose=False):

    """
    This function solves directly the spectral solution (Gauss collocation formulation). That is,
    this function solves

    .. math::
        y - \\Delta{t}SF(y) = y_0

    :param function f_eval: the derivative function :math:`y' = f(t,y)`.
    :param numpy.ndarray t: the time nodes over a time step :math:`\\Delta{t}`
    :param numpy.ndarray y0: the initial solutions length, size of the problem
    :param numpy.ndarray S: the spectral integration matrix :math:`S`
    :param float f_tol: the relative tolerance of the Newton-Krylov solver
    :param numpy.ndarray y_approx: the initial guess for the Newton-Krylov solver
    :param bool verbose: a flag for the built-in Newton-Krylov solver

    :return: the spectral solution
    :rtype: numpy.ndarray
    """

    # start timing
    start = time.time()

    # the number of time nodes
    n_nodes = S.shape[1]

    # copy the initial condition [n_nodes x m]
    y0_vec  = np.vstack([y0 for _ in range(n_nodes)])

    # solve H(y) = y - SF(y) -y0_vec = 0
    H       = lambda y: spectral_root_finder(f_eval, t, y, y0_vec, S)

    # get the solution
    if y_approx is None:
        y_approx = y0_vec

    y   = newton_krylov(H, y_approx, f_rtol=f_tol, verbose=verbose)
    #y = newton_krylov(H, y_approx, f_tol=f_tol, verbose=verbose)
    #y   = root(H, y_approx, tol=f_tol, method='krylov').x
    #y   = anderson(H, y_approx)

    # end timing
    end = time.time()

    if do_print:
        print_elapsed_time(start, end)

    return y

def spectral_root_finder(f_eval, t, y, y0_vec, S):


    """
    This function calculates the residual in the spectral (Gauss) collocation formulation. That is,

    .. math::
        y - \\Delta{t}SF(t,y) - y_0.

    It is used in the root finding algorithm to solve

    .. math::
        A(y) = y - \\Delta{t}SF(t,y) - y_0 = 0
    
    :param function f_eval: the derivative function :math:`y' = f(t,y)`.
    :param numpy.ndarray t: the time nodes
    :param numpy.ndarray y: an approximate solution [number of temporal nodes x size of problem]
    :param y0_vec: the initial condition vector [number of temporal nodes x size of problem]
    :param numpy.ndarray S: the spectral integration (Gauss quadrature) matrix :math:`S`

    :return: the residual in the spectral collocation formulation
    :rtype:  numpy.ndarray  [ number of temporal nodes x size of problem ]
    """

    # the number of temporal nodes, the size of the system
    n_nodes, m = y0_vec.shape

    # the derivative function
    F       = np.zeros(y0_vec.shape)
    temp    = np.zeros(y0_vec.shape)

    # calculate the derivative at all nodes
    for i in range(n_nodes):
        F[i] = f_eval(t[i], y[i])

    # spectral integration
    for i in range(m):
        temp[:, i] = S.dot(F[:, i])

    # the residual
    res = y - temp - y0_vec

    return res

def step_size_scaler(y, ysteps, n_nodes, k, tol, node_type):

    """
    This function

    :param numpy.ndarray y: an approximation of the ODE system using 1 step size of :math:`h=\\Delta{t}`
    :param numpy.ndarray ysteps: an approximation of the ODE system using :math:`k` steps of size \
    :math:`h=\\frac{\\Delta{t}}{k}`
    :param int n_nodes: the number of temporal nodes within the time step
    :param int k: the amount of mini time steps leading up to :math:`y(\\Delta{t})`
    :param float tol: the absolute error tolerance wanted within the time step
    :param int node_type: the type of temporal nodes

    Given a step size :math:`\\Delta{t}`, calculate :math:`x` the amount the step size should increase \
    or decrease for the proper adaptive time step size :math:`\\Delta{t}_{new}` where

    .. math::
        \\Delta{t}_{new} = x \\Delta{t}

    .. note::
        This function currently runs using only Gauss-Lobatto nodes

    :return:
    """

    # this is for Gauss-Lobatto
    assert node_type == points.GAUSS_LOBATTO

    # aerr is the maximum error at time dt
    aerr = LA.norm(y - ysteps, ord=np.inf)

    # k is the number of substeps to get to stepsize dt
    if node_type == points.GAUSS_LOBATTO:
        scaler = scaler_lobatto(aerr, n_nodes, k, tol)

    return scaler

def stopping_criteria(t_final, t0):

    """
    This function sends a flag whether or not we have reached the end of the simulation while \
    taking account errors from inexact arithmetic.

    :param float t_final: final time in the ODE simulation
    :param float t0: the start time for the current time step

    :return: a flag indicating whether or not we have reached the end of the simulation
    :rtype: bool
    """

    # the criterion the final time being "small".
    T_SMALL = 1e-15

    # this is to make up for errors due to inexact arithmetic
    EPS = 1e-20

    # the time taking into account inexact arithmetic error
    t = t0 + EPS

    # t_final is considered "big", so use absolute difference
    # or t_final is 0, avoid division by 0
    if (t_final >= T_SMALL) or (t_final == 0):
        do_stop = t >= t_final
    else:
        # t_final is considered "small", use the relative difference
        # this is true if t > t_final or abs(t)/t_final < respective magnitude
        do_stop = (t_final - t) / t_final <= 1e-5

    return do_stop

def update_step_size(dt, t0, t_final):

    """
    This function makes sure that the time step :math:`\\Delta{t}` does not cause the simulation to go \
    past the final time :math:`t_{final}`.

    :param float dt: the current step size :math:`\\Delta{t}`
    :param float t0: the start time for the current time step
    :param float t_final: the final time :math:`t_{final}` of the ODE simulation

    :return: the step size for the next time step
    :rtype: float
    """

    # assume that
    dt_new = dt

    # do set the step size so that it ends at t_final
    if (t0 + dt) > t_final:
        dt_new = t_final - t0

    return dt_new