#
# Copyright 2018, 2020 Antoine Sanner
#           2019 Lars Pastewka
#
# ### MIT license
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#

import numpy as np

from .. import MPI


def get_dtype_info(dtype):
    """
    Get the machine limits for integer or floating point types.

    Parameters
    ----------
    dtype : numpy.dtype
        The data type to get information about.

    Returns
    -------
    info : numpy.iinfo or numpy.finfo
        Object containing machine limits for the given dtype.
        For integer types, returns numpy.iinfo.
        For floating point types, returns numpy.finfo.

    Raises
    ------
    ValueError
        If the dtype is neither integer nor floating point.
    """
    if dtype.kind == 'i':
        return np.iinfo(dtype)
    if dtype.kind == 'f':
        return np.finfo(dtype)
    raise ValueError(f"Unsupported dtype kind '{dtype.kind}': expected 'i' (integer) or 'f' (float)")


class Reduction:
    """
    MPI-parallel reduction operations for numpy arrays.

    This class provides a unified interface for performing reduction operations
    (sum, max, min, mean, dot product, etc.) that work transparently in both
    serial and MPI-parallel contexts. Each method performs a local numpy
    operation followed by an MPI Allreduce to combine results across all
    processes.

    When used in serial mode (no MPI communicator provided), it uses
    MPI.COMM_SELF which effectively performs no communication, making the
    code work identically in serial and parallel contexts.

    Parameters
    ----------
    comm : MPI.Comm, optional
        MPI communicator. If None, uses MPI.COMM_SELF for serial operation.

    Attributes
    ----------
    comm : MPI.Comm
        The MPI communicator used for collective operations.

    Examples
    --------
    Serial usage (no MPI):

    >>> import numpy as np
    >>> from NuMPI.Tools import Reduction
    >>> pnp = Reduction()
    >>> arr = np.array([1.0, 2.0, 3.0])
    >>> pnp.sum(arr)
    array(6.)

    Parallel usage (with MPI):

    >>> from mpi4py import MPI
    >>> comm = MPI.COMM_WORLD
    >>> pnp = Reduction(comm)
    >>> local_arr = np.array([1.0, 2.0])  # each process has different data
    >>> global_sum = pnp.sum(local_arr)   # sum across all processes
    """

    def __init__(self, comm=None):
        """
        Initialize a Reduction instance.

        Parameters
        ----------
        comm : MPI.Comm, optional
            MPI communicator for collective operations. If None, defaults to
            MPI.COMM_SELF which performs no inter-process communication,
            enabling the same code to run in serial mode.
        """
        if comm is None:
            self.comm = MPI.COMM_SELF
        else:
            self.comm = comm

    def _op(self, npop, npargs, mpiop, *args, **kwargs):
        """
        Perform a generic MPI-parallel reduction operation.

        This is an internal method that handles the pattern of:
        1. Apply a numpy reduction function locally
        2. Combine results across all processes using MPI Allreduce

        Parameters
        ----------
        npop : callable
            Numpy reduction function (e.g., np.sum, np.dot).
        npargs : tuple
            Positional arguments passed to the numpy function. Typically
            contains the array(s) to be reduced.
        mpiop : MPI.Op
            MPI reduction operation (e.g., MPI.SUM, MPI.MAX, MPI.MIN).
        *args : tuple
            Additional positional arguments passed to npop.
        **kwargs : dict
            Keyword arguments passed to npop (e.g., axis, keepdims).

        Returns
        -------
        result : np.ndarray
            Result of the global reduction operation, available on all processes.
        """
        local_result = npop(*npargs, *args, **kwargs)
        result = np.zeros_like(local_result)
        mpitype = MPI._typedict[local_result.dtype.char]
        self.comm.Allreduce([local_result, mpitype], [result, mpitype], op=mpiop)
        return result

    def _op1(self, npop, arr, mpiop, *args, **kwargs):
        """
        Perform a generic MPI-parallel reduction on a single array.

        This is a convenience wrapper around `_op` for operations that take
        a single array argument. It also handles special cases for masked
        arrays where the `initial` parameter is not supported.

        Parameters
        ----------
        npop : callable
            Numpy reduction function (e.g., np.sum, np.max, np.min).
        arr : array_like
            Numpy array containing the local data to be reduced.
        mpiop : MPI.Op
            MPI reduction operation (e.g., MPI.SUM, MPI.MAX).
        *args : tuple
            Additional positional arguments passed to npop.
        **kwargs : dict
            Keyword arguments passed to npop. The `initial` keyword is
            handled specially for masked arrays.

        Returns
        -------
        result : np.ndarray
            Result of the global reduction operation.

        Notes
        -----
        For masked arrays with an `initial` value, the masked values are
        filled with the initial value before reduction, since numpy's
        max/min on masked arrays do not support the `initial` parameter.
        """
        if 'initial' in kwargs and isinstance(arr, np.ma.MaskedArray):
            # Max/min on masked array do not support `initial`
            arr = arr.filled(kwargs['initial'])
            del kwargs['initial']
        return self._op(npop, (arr,), mpiop, *args, **kwargs)

    def sum(self, arr, *args, **kwargs):
        """
        Compute the global sum of a distributed array.

        Sums all elements across all MPI processes. Each process contributes
        its local sum, and the global sum is returned to all processes.

        Parameters
        ----------
        arr : array_like
            Numpy array containing the local data to be summed.
        *args : tuple
            Additional positional arguments passed to numpy.sum.
        **kwargs : dict
            Keyword arguments passed to numpy.sum (e.g., axis, keepdims).

        Returns
        -------
        result : np.ndarray
            Global sum of all elements across all processes.
        """
        return self._op1(np.sum, arr, MPI.SUM, *args, **kwargs)

    def max(self, arr, *args, **kwargs):
        """
        Compute the global maximum of a distributed array.

        Finds the maximum value across all elements on all MPI processes.

        Parameters
        ----------
        arr : array_like
            Numpy array containing the local data.
        *args : tuple
            Additional positional arguments passed to numpy.max.
        **kwargs : dict
            Keyword arguments passed to numpy.max (e.g., axis, keepdims).

        Returns
        -------
        result : np.ndarray
            Global maximum value across all processes.

        Notes
        -----
        Uses an initial value of dtype.min to handle empty local arrays
        correctly (an empty array contributes the smallest possible value).
        """
        kwargs['initial'] = get_dtype_info(arr.dtype).min
        return self._op1(np.max, arr, MPI.MAX, *args, **kwargs)

    def min(self, arr, *args, **kwargs):
        """
        Compute the global minimum of a distributed array.

        Finds the minimum value across all elements on all MPI processes.

        Parameters
        ----------
        arr : array_like
            Numpy array containing the local data.
        *args : tuple
            Additional positional arguments passed to numpy.min.
        **kwargs : dict
            Keyword arguments passed to numpy.min (e.g., axis, keepdims).

        Returns
        -------
        result : np.ndarray
            Global minimum value across all processes.

        Notes
        -----
        Uses an initial value of dtype.max to handle empty local arrays
        correctly (an empty array contributes the largest possible value).
        """
        kwargs['initial'] = get_dtype_info(arr.dtype).max
        return self._op1(np.min, arr, MPI.MIN, *args, **kwargs)

    def mean(self, arr, *args, **kwargs):
        """
        Compute the global arithmetic mean of a distributed array.

        Calculates the mean across all elements on all MPI processes by
        computing the global sum divided by the global count.

        Parameters
        ----------
        arr : array_like
            Numpy array containing the local data.
        *args : tuple
            Additional positional arguments passed to the underlying sum.
        **kwargs : dict
            Keyword arguments passed to the underlying sum.

        Returns
        -------
        result : np.ndarray
            Global arithmetic mean across all processes.

        Notes
        -----
        This correctly handles unequal data distribution across processes
        by computing sum(arr) / sum(ones_like(arr)) rather than averaging
        local means.
        """
        return self.sum(arr, *args, **kwargs) / self.sum(np.ones_like(arr), *args, **kwargs)

    def dot(self, a, b, *args, **kwargs):
        """
        Compute the global scalar (dot) product of two distributed arrays.

        Computes the dot product a · b across all MPI processes. Each process
        computes its local dot product, then results are summed globally.

        Parameters
        ----------
        a : array_like
            First input array (local portion on this process).
        b : array_like
            Second input array (local portion on this process). Must have
            the same shape as `a`.

        Returns
        -------
        result : np.ndarray
            The global scalar product of a and b, summed across all processes.
            Returns a scalar (0-d array).

        Notes
        -----
        For distributed arrays where each process holds a portion of the full
        arrays, this computes the mathematically correct global dot product
        by summing local contributions.
        """
        return self._op(np.dot, (a, b), MPI.SUM, *args, **kwargs)

    def any(self, arr, *args, **kwargs):
        """
        Test whether any element is True across all MPI processes.

        Performs a global logical OR operation: returns True if any element
        in any process's local array evaluates to True.

        Parameters
        ----------
        arr : array_like
            Input array. Will be cast to bool for the test.
        *args : tuple
            Additional positional arguments passed to numpy.any.
        **kwargs : dict
            Keyword arguments passed to numpy.any (e.g., axis).

        Returns
        -------
        result : np.ndarray
            Boolean result. True if any element across all processes is True.

        Notes
        -----
        Uses MPI logical OR (MPI.LOR) to combine local results.
        """
        return self._op1(np.any, arr, MPI.LOR, *args, **kwargs)

    def all(self, arr, *args, **kwargs):
        """
        Test whether all elements are True across all MPI processes.

        Performs a global logical AND operation: returns True only if all
        elements in all processes' local arrays evaluate to True.

        Parameters
        ----------
        arr : array_like
            Input array. Will be cast to bool for the test.
        *args : tuple
            Additional positional arguments passed to numpy.all.
        **kwargs : dict
            Keyword arguments passed to numpy.all (e.g., axis).

        Returns
        -------
        result : np.ndarray
            Boolean result. True only if all elements across all processes
            are True.

        Notes
        -----
        Uses MPI logical AND (MPI.LAND) to combine local results.
        """
        return self._op1(np.all, arr, MPI.LAND, *args, **kwargs)
