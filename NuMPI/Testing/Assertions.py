import numpy as np

from ..Tools.Reduction import Reduction


def _nb_failed(comm, failed):
    """Number of ranks on which ``failed`` is true. Through ``Reduction``, so
    any communicator NuMPI accepts works (mpi4py's or one implementing the
    ``allreduce`` protocol, e.g. muGrid's)."""
    return int(Reduction(comm).sum(np.array([failed], dtype=np.int32)))


def parallel_assert(comm, condition, message="Assertion failed"):
    if _nb_failed(comm, not condition) > 0:
        raise AssertionError(message)


def _assert_one(fun):
    def assert_func(comm, rank, *args, **kwargs):
        exception = None
        if comm.rank == rank:
            try:
                fun(*args, **kwargs)
            except AssertionError as exc:
                exception = exc

        if _nb_failed(comm, exception is not None) > 0:
            if exception is None:
                raise AssertionError()
            else:
                raise exception

    return assert_func


def _assert_all(fun):
    def assert_func(comm, *args, **kwargs):
        exception = None
        try:
            fun(*args, **kwargs)
        except AssertionError as exc:
            exception = exc

        if _nb_failed(comm, exception is not None) > 0:
            if exception is None:
                raise AssertionError()
            else:
                raise exception

    return assert_func


assert_one_array_equal = _assert_one(np.testing.assert_array_equal)
assert_all_array_equal = _assert_all(np.testing.assert_array_equal)

assert_one_allclose = _assert_one(np.testing.assert_allclose)
assert_all_allclose = _assert_all(np.testing.assert_allclose)
