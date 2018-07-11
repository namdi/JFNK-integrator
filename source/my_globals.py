# This file was written by Dr. Namdi Brandon
# ORCID: 0000-0001-7050-1538
# June 29, 2018

"""
This module contains some functions and constants that are useful for global use.

.. moduleauthor:: Dr. Namdi Brandon
"""

# ===============================================
# import
# ===============================================
import os, pickle

# ===============================================
# constants
# ===============================================
# the file path to store data
FPATH_MY_DATA = '..\\my_data'

# ===============================================
# function
# ===============================================

def load(fname):

    """
    This function loads data from a .pkl file.

    :param str fname: the file name to be loaded from
    :return: the data unpickled
    """

    # open the file for reading
    fin = open(fname, 'rb')

    # load the data
    x = pickle.load(fin)

    # close the file
    fin.close()

    return x

def save(x, fname):

    """
    This function saves a python variable by pickling it.

    :param x: the data to be saved
    :param str fname: the file name of the saved file. It must end with .pkl

    """

    # create the directory for the save file if it does not exist
    os.makedirs(os.path.dirname(fname), exist_ok=True)

    # open the file for writing
    fout = open(fname, 'wb')

    # save the file as a binary
    pickle.dump(x, fout)

    # close the file
    fout.close()

    return