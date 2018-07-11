# This file was written by Dr. Namdi Brandon
# ORCID: 0000-0001-7050-1538
# June 29, 2018

"""
This module contains functions necessary for doing prepossessing before solving the
differential equation system.

This module contains functions for the following

* temporal nodes in :math:`t \\in [0, 1]`
* backward Euler matrix
* spectral integration matrix
* spectral radius

.. moduleauthor:: Dr. Namdi Brandon
"""

# ===============================================
# import modules
# ===============================================

import numpy as np
import numpy.linalg as LA

# for the numerical integrator
from scipy import integrate

# this comes from the pyweno code from Matt Emmett
import points

# ===============================================
# functions
# ===============================================

def backward_euler_matrix(t):

    """
    This function calculates the backward Euler integration matrix

    :math:`\\left(\\tilde{S}\\right)` assuming :math:`t \\in [0, 1]`. That is, given
    :math:`t=[t_0, t_1, \\ldots, t_{n-1}]` we have
    :math:`0 \\le t_0 < t_1 < \\ldots t_{n-1} \\le 1`.

     .. math::

        \\tilde{S} =
        \\begin{bmatrix}
            \\Delta{t}_0 & 0 & \\cdots & 0 & 0 \\\\
            \\Delta{t}_0 & \\Delta{t}_1 & \\cdots & 0 & 0 \\\\
            \\cdot & \\cdot & \\cdots & 0 & 0  \\\\
            \\Delta{t}_0 & \\Delta{t}_1 & \\cdots & \\Delta{t}_{n-3} & 0 \\\\
            \\Delta{t}_0 & \\Delta{t}_1 & \\cdots & \\Delta{t}_{n-3} & \\Delta{t}_{n-2}
        \\end{bmatrix}

    where
        .. math::

            \\begin{cases}
                \\Delta{t}_0 &= t_0 - 0 \\\\
                \\Delta{t}_i &= t_i - t_{i-1} \,\, i=1, \\ldots, n-1
            \\end{cases}

    :param np.ndarray t: the time nodes (length n)

    :return: the backward Euler integration matrix :math:`\\tilde{S}`
    :rtype: numpy.ndarray
    """

    # is the end point included in the nodes
    do_end_point = t[-1] == 1.0

    if not do_end_point:
        tau = np.hstack( [t, 1] )
    else:
        tau = np.array(t)

    # the length of the time nodes
    n = len(tau)

    # the integration matrix
    S = np.zeros((n, n))

    # for each column in S
    for c in range(n):

        if (c == 0):
            S[:, c] = tau[c] - 0
        else:
            S[c:, c] = tau[c] - tau[c - 1]

    return S


def gauss_lobatto(n):

    """
    This function returns Gauss-Lobatto nodes :math:`t \\in [0, 1]`.

    :param n: the number of nodes
    :return: the Gauss-Lobatto nodes with :math:`t \\in [0, 1]`
    :rtype: numpy.ndarray
    """

    # return the nodes in [-1, 1]
    x = points.gauss_lobatto(n)

    # convert the values to a float and store them in an array
    x = np.array( [ float(u) for u in x] )

    # shift and scale the nodes to be in [0, 1]
    x = (x + 1) / 2.0

    # reshape into a column vector
    x.resize( (n, 1) )

    return x

def gauss_lobatto_error_constant(n):

    """
    This is the constant in the error term :math:`(R_n)` for Gauss-Lobatto quadrature. That is,

    .. math::
        \\| \int^{1}_{0}f(\\tau) \\, \\mathrm{d}\\tau - \\sum^{n-1}_{j=0} w_jf(t_j) \\| = R_n

    where

    .. math::
        R_n &= -\\frac{ \\displaystyle {n(n-1)^3[(n-2)!]^4} }{ {(2n-1)[(2n-2)!]^3} } \\Delta{t}^{2n-1} f^{(2n-2)}(\\tau), \\text{\\indent} 0 < \\tau < 1\\\\
        &= -c_n \\Delta{t}^{2n-1} f^{(2n-2)}(\\tau)

    :param n: the number of nodes
    :return: the constant :math:`c` in the error term :math:`R_n`
    """

    factorial = np.math.factorial

    # the numerator
    top = n * (n - 1)** 3 * factorial(n - 2)**4

    # the denominator
    bot = (2 * n - 1) * factorial(2 * n - 2)**3

    # the constant
    c   = top / bot

    return c

def get_nodes(n , node_type):

    """
    This function various types of nodes :math:`t \\in [0, 1]`. Currently,
    this function may calculate the following types of nodes:

    * Gauss-Legendre nodes
    * Gauss-Lobatto nodes
    * Guass Radau nodes (left end point)
    * Gauss Radau 2a nodes (right end point)

    :param int n: the number of nodes
    :param int node_type: the spectral node type
    :return: the spectral nodes in :math:`t \\in [0, 1]`
    """

    # return the nodes in [-1, 1]
    if node_type == points.GAUSS_LEGENDRE:
        x = points.gauss_legendre(n)

    elif node_type == points.GAUSS_LOBATTO:
        x = points.gauss_lobatto(n)

    elif node_type == points.GAUSS_RADAU:
        x = points.gauss_radau(n)

    elif node_type == points.GAUSS_RADAU_2A:
        x = points.gauss_radau_2a(n)

    # convert the values to a float and store them in an array
    x = np.array([float(u) for u in x])

    # shift and scale the nodes to be in [0, 1]
    x = (x + 1) / 2.0

    return x

def spectral_matrix(t):

    """
    This function calculates the spectral integration (Gaussian quadrature) matrix applied to the
    time step :math:`\\Delta{t}` where each entry is of :math:`S` is given by

    .. math::
        S_{ij}  = \\int^{t_i}_{0} \\left( \\prod_{k \\neq j}  \\frac{t - t_k}{t_j - t_k}\\right) \, \\mathrm{d}t

    :param numpy.ndarray t: the temporal nodes :math:`t \\in [0, 1]`. length n

    :return: the spectral integration matrix
    :rtype: numpy.ndarray
    """

    # the length of the time nodes
    n = len(t)

    # is one of the nodes and right hand end point
    do_end_point = t[-1] == 1.0

    # the spectral integration matrix
    if do_end_point:
        S = np.zeros( (n, n) )
    else:
        S = np.zeros( (n+1, n+1) )

    # for each column in S
    for c in range(n):

        # the legendre interpolation polynomial weight function
        f = lambda x: np.prod( [ (x - u) for u in t if u != t[c] ] )
        denom = np.prod( [ (t[c] - u) for u in t if u != t[c] ] )

        # integrate the legendre weight function over different times
        for r in range(n):
            (result, aerr) = integrate.quad(f, 0, t[r]) / denom
            S[r,c] = result

        if not do_end_point:
            (result, aerr) = integrate.quad(f, 0, 1) / denom
            S[n, c] = result

    return S

def spectral_radius(node_type, S, S_p):

    """
    This function calculates the spectral radius (the the largest magnitude of the eigenvalue) of the
    SDC correction matrix for the extremely stiff case.

    First, calculate the correction matrix :math:`C` for the extremely stiff case

    .. math::
        \\rho(C) = \\rho(I - \\tilde{S}^{-1}S)

    :param int node_type: the type of node points
    :param S: the spectral integration matrix :math:`S`
    :param S_p: the preconditioner integration matrix :math:`\\tilde{S}`

    :return: the spectral radius
    :rtype: float
    """

    # the number of time nodes used in the spectral integration
    n = S.shape[0]

    # the correction matrix
    C = None

    # form the correction matrix in the stiff case I - S_p^{-1} S
    if node_type == points.GAUSS_LEGENDRE:
        C = np.eye(n-1) - LA.inv( S_p[:-1, :-1] ).dot( S[:-1, :-1] )

    elif node_type == points.GAUSS_LOBATTO:
        C = np.eye(n-1) - LA.inv(S_p[1:, 1:] ).dot( S[1:, 1:])

    elif node_type == points.GAUSS_RADAU:
        C = np.eye(n-1) - LA.inv ( S_p[1:, 1:] ).dot( S[1:, :-1] )

    elif node_type == points.GAUSS_RADAU_2A:
        C = np.eye(n) - LA.inv(S_p).dot(S)

    # eigen decomposition of the correction matrix
    e_vals, e_vecs = LA.eig(C)

    # the spectral radius
    radius = np.abs(e_vals).max()

    return radius

