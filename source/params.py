# This file was written by Dr. Namdi Brandon
# ORCID: 0000-0001-7050-1538
# June 29, 2018

"""
This module contains :class:`params.Params` which contains information related to
the differential equation system that is beings solved.
"""

# ===============================================
# import
# ===============================================

import points
import preprocess as pre

# ===============================================
# class
# ===============================================-

class Params(object):

    """
    This class includes information related to the differential equation system solver.

    :param int n_nodes: the number of nodes in a time step
    :param int m: the size of the system
    :param node_type: the type of nodes used in the tine step

    :var int n_nodes: the number of temporal nodes
    :var int m: the size of the problem
    :var nunpy.ndarray t: the time nodes at such that :math:`t \\in [0, 1]`
    :var numpy.ndarray S: the spectral integration (Gauss quadrature) matrix
    :var numpy.ndarray S_p: the preconditioner integration matrix
    :var float 'spectral_radius': the spectral radius of the correction matrix from spectral \
    deferred correction matrix.
    :var float c: the error constant in the error term from Gauss quadrature and Gauss-Lobatto \
    nodes
    """

    def __init__(self, n_nodes, m, node_type=points.GAUSS_LOBATTO):

        # number of nodes in 1 time step
        self.n_nodes    = n_nodes

        # size of the problem
        self.m          = m

        # the type of nodes
        self.node_type  = node_type

        # the node points in [0, 1]
        self.t      = pre.get_nodes(self.n_nodes, self.node_type)

        # spectral integration matrix (normalized)
        self.S      =  pre.spectral_matrix(self.t)

        # preconditioner integration matrix (normalized)
        self.S_p    =  pre.backward_euler_matrix(self.t)

        # the spectral radius of the correction matrix in the extremely stiff case
        self.spectral_radius = pre.spectral_radius(self.node_type, self.S, self.S_p)

        # the constant in the error term in integration c(n) * f^(2n-2)(x) * dt**(2n-1)
        self.c = pre.gauss_lobatto_error_constant(self.n_nodes)

        return


