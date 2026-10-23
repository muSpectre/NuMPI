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


# Communicators that are not mpi4py's (e.g. muGrid's ``Communicator``, which
# does not need mpi4py) are driven through a small duck-typed protocol:
#
#   comm.allreduce(array, op)  op in "sum" | "max" | "min"; returns an array of
#                              the same shape and dtype on every rank
#   comm.compare(other)        "identical" | "congruent" | "similar" | "unequal"
#   comm.size, comm.rank
#
# A communicator with mpi4py's buffer method ``Allreduce`` takes the mpi4py
# path instead; this is how the two are told apart.

# dtypes every protocol communicator reduces; others are reduced in a wider
# type of the same kind and cast back
_PROTOCOL_DTYPES = {np.dtype(t) for t in (np.float32, np.float64, np.complex64,
                                          np.complex128, np.int32, np.int64,
                                          np.uint32)}


def _is_mpi4py_like(comm):
    """Whether ``comm`` is driven through mpi4py's buffer interface (mpi4py
    itself and NuMPI's MPI stub)."""
    return hasattr(comm, "Allreduce")


def comm_size(comm):
    """Number of ranks of an mpi4py-like or protocol communicator."""
    return comm.Get_size() if hasattr(comm, "Get_size") else comm.size


def _protocol_allreduce(comm, local, op):
    """Reduce ``local`` (array or scalar) with a protocol communicator,
    returning an array of the local result's shape and dtype."""
    local = np.asarray(local)
    dtype = local.dtype
    if dtype == np.bool_:
        # any/all: logical or/and are max/min on 0/1
        work = local.astype(np.int32)
    elif dtype in _PROTOCOL_DTYPES:
        work = local
    elif dtype.kind in "iub":
        work = local.astype(np.int64)
    elif dtype.kind == "c":
        work = local.astype(np.complex128)
    else:
        work = local.astype(np.float64)
    # Sent flat: a 0-d array (the common scalar sum) is a one-element buffer
    flat = np.ascontiguousarray(work).reshape(-1)
    result = np.asarray(comm.allreduce(flat, op))
    return result.astype(dtype, copy=False).reshape(local.shape)


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
    comm : MPI.Comm or communicator, optional
        MPI communicator. If None, uses MPI.COMM_SELF for serial operation.
        Besides mpi4py communicators, any communicator implementing
        ``allreduce(array, op)`` with ``op`` one of ``"sum"``, ``"max"``,
        ``"min"`` (and ``compare``, ``size``, ``rank``) is accepted, such as
        muGrid's ``Communicator``, which does not need mpi4py.

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
        mpiop : str
            Reduction: ``"sum"``, ``"max"`` or ``"min"`` (``MPI.SUM``,
            ``MPI.MAX``, ``MPI.MIN`` for mpi4py communicators).
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
        if not _is_mpi4py_like(self.comm):
            return _protocol_allreduce(self.comm, local_result, mpiop)
        result = np.zeros_like(local_result)
        mpitype = MPI._typedict[local_result.dtype.char]
        self.comm.Allreduce([local_result, mpitype], [result, mpitype],
                            op=getattr(MPI, mpiop.upper()))
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
        mpiop : str
            Reduction: ``"sum"``, ``"max"`` or ``"min"``.
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
        return self._op1(np.sum, arr, "sum", *args, **kwargs)

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
        return self._op1(np.max, arr, "max", *args, **kwargs)

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
        return self._op1(np.min, arr, "min", *args, **kwargs)

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
        return self._op(np.dot, (a, b), "sum", *args, **kwargs)

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
        Reduced as the maximum of 0/1 (int32), which every communicator
        supports, rather than as a logical OR.
        """
        return self._logical(np.any, arr, "max", *args, **kwargs)

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
        Reduced as the minimum of 0/1 (int32), which every communicator
        supports, rather than as a logical AND.
        """
        return self._logical(np.all, arr, "min", *args, **kwargs)

    def _logical(self, npop, arr, op, *args, **kwargs):
        """``any``/``all``: the local test as 0/1 in int32, reduced with
        ``op`` ("max" or "min") and returned as bool."""
        def local(a, *args, **kwargs):
            return np.asarray(npop(a, *args, **kwargs)).astype(np.int32)

        return self._op(local, (arr,), op, *args, **kwargs).astype(bool)


def _comm_of(pnp):
    """
    Underlying communicator of a reduction wrapper.

    A plain ``numpy`` reduction (the serial fallback) has no communicator and
    is mapped to ``MPI.COMM_SELF``, i.e. a single-rank group, which is exactly
    its reduction semantics.
    """
    if pnp is np:
        return MPI.COMM_SELF
    comm = getattr(pnp, "comm", None)
    return MPI.COMM_SELF if comm is None else comm


def same_reduction_group(pnp_a, pnp_b):
    """
    Return ``True`` if two reduction wrappers reduce over the same group of
    MPI ranks.

    Comparing the wrapper objects directly (``pnp_a is pnp_b`` or ``==``) is
    *not* a reliable check: a solver handed a bare ``comm`` builds its own
    ``Reduction(comm)`` internally, so even correct usage produces two distinct
    ``Reduction`` instances wrapping the same communicator. Correctness depends
    only on the communicator's process group, not on object identity, so we
    compare the communicators with ``MPI.Comm.Compare`` and accept ``IDENT`` /
    ``CONGRUENT`` (identical group membership); communicators that are not
    mpi4py's are compared with their own ``compare``, which reports
    ``"identical"`` / ``"congruent"`` alike.

    A ``numpy`` reduction is treated as a single-rank communicator; if both
    sides reduce over a single rank they are trivially compatible. This also
    covers the serial MPI stub, which is always size 1 and has no ``Compare``.

    Parameters
    ----------
    pnp_a, pnp_b : module-like
        Reduction wrappers to compare (``Reduction`` instances or ``numpy``).

    Returns
    -------
    bool
    """
    comm_a = _comm_of(pnp_a)
    comm_b = _comm_of(pnp_b)
    # Same communicator object: unambiguously compatible.
    if comm_a is comm_b:
        return True
    # Both reduce over a single rank (numpy and/or the serial stub): the
    # reduction is the identity, so the group is immaterial.
    if comm_size(comm_a) == 1 and comm_size(comm_b) == 1:
        return True
    # A protocol communicator (e.g. muGrid's) compares itself, also with an
    # mpi4py communicator over the same MPI library.
    for comm, other in ((comm_a, comm_b), (comm_b, comm_a)):
        if not _is_mpi4py_like(comm) and hasattr(comm, "compare"):
            return comm.compare(other) in ("identical", "congruent")
    # Real mpi4py: compare process groups. `Comm.Compare` is a local
    # operation, so this is safe to call outside a collective context.
    comm_cls = getattr(MPI, "Comm", None)
    compare = getattr(comm_cls, "Compare", None)
    if compare is None:
        # MPI stub exposes no `Comm`/`Compare`; every stub communicator is size
        # 1 and is handled above. Fall back to a conservative size comparison.
        return comm_size(comm_a) == comm_size(comm_b)
    return compare(comm_a, comm_b) in (MPI.IDENT, MPI.CONGRUENT)
