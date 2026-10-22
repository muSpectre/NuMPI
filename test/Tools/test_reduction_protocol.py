#
# Copyright 2026 Lars Pastewka
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
"""
Reductions through communicators that are not mpi4py's: a small duck-typed
protocol (``allreduce(array, op)`` with ``op`` one of "sum", "max", "min";
``compare``; ``size``; ``rank``), as implemented by muGrid's ``Communicator``.

The protocol is exercised twice: with a serial fake communicator (always
runs) and with muGrid's ``Communicator.world()`` over all ranks of the
launcher (skipped without a muGrid that provides it).
"""

import numpy as np
import pytest

from NuMPI.Optimization import l_bfgs_bounded
from NuMPI.Testing.Assertions import assert_all_allclose, parallel_assert
from NuMPI.Tools import Reduction
from NuMPI.Tools.Reduction import same_reduction_group


class FakeComm:
    """Serial communicator implementing only the protocol. Records the
    buffers it was asked to reduce."""

    size = 1
    rank = 0

    def __init__(self):
        self.calls = []

    def allreduce(self, a, op):
        assert op in ("sum", "max", "min")
        assert isinstance(a, np.ndarray) and a.ndim == 1
        self.calls.append((a.dtype, op))
        return a.copy()

    def compare(self, other):
        return "identical" if other is self else "unequal"


def _mugrid_world():
    muGrid = pytest.importorskip("muGrid")
    world = getattr(muGrid.Communicator, "world", None)
    if world is None:
        pytest.skip("muGrid without Communicator.world()")
    return world()


@pytest.fixture(params=["fake", "muGrid"])
def comm(request):
    if request.param == "fake":
        return FakeComm()
    return _mugrid_world()


def test_fake_comm_is_not_driven_through_mpi4py():
    c = FakeComm()
    Reduction(c).sum(np.arange(3.0))
    assert c.calls == [(np.dtype(np.float64), "sum")]


@pytest.mark.parametrize("dtype, sent", [
    (np.float64, np.float64), (np.float32, np.float32),
    (np.int64, np.int64), (np.int32, np.int32),
    (np.int16, np.int64), (np.int8, np.int64), (np.float16, np.float64),
])
def test_fake_comm_dtypes(dtype, sent):
    """dtypes the protocol does not promise are reduced in a wider type of
    the same kind, and come back in their own dtype. (Through ``max``: numpy's
    ``sum`` already promotes small integers itself.)"""
    c = FakeComm()
    res = Reduction(c).max(np.arange(4, dtype=dtype))
    assert c.calls == [(np.dtype(sent), "max")]
    assert res.dtype == np.dtype(dtype) and res == 3


def test_fake_comm_any_all_are_int32_max_min():
    c = FakeComm()
    pnp = Reduction(c)
    assert pnp.any(np.array([False, True])) and not pnp.all(np.array([False, True]))
    assert c.calls == [(np.dtype(np.int32), "max"), (np.dtype(np.int32), "min")]


def test_sum_scalar(comm):
    res = Reduction(comm).sum(np.array(1))
    assert res.shape == () and res == comm.size


def test_sum_python_int(comm):
    assert Reduction(comm).sum(3) == 3 * comm.size


def test_sum_axis_keeps_shape(comm):
    a = np.arange(6.0).reshape(2, 3) + comm.rank
    res = Reduction(comm).sum(a, axis=0)
    expected = comm.size * np.arange(6.0).reshape(2, 3).sum(axis=0) \
        + 2 * comm.size * (comm.size - 1) / 2
    assert res.shape == (3,)
    np.testing.assert_allclose(res, expected)


def test_max_min_mean_dot(comm):
    pnp = Reduction(comm)
    a = np.array([1.0, 2.0]) + comm.rank
    assert pnp.max(a) == 2.0 + comm.size - 1
    assert pnp.min(a) == 1.0
    assert pnp.mean(a) == pytest.approx(1.5 + (comm.size - 1) / 2)
    assert pnp.dot(a, a) == pytest.approx(
        sum((1 + r) ** 2 + (2 + r) ** 2 for r in range(comm.size)))


def test_max_of_empty_local_array(comm):
    """An empty slab contributes the dtype's minimum (``initial``)."""
    a = np.array([] if comm.rank > 0 else [3.0])
    assert Reduction(comm).max(a) == 3.0


def test_complex_sum(comm):
    res = Reduction(comm).sum(np.array([1 + 2j]))
    assert res.dtype == np.complex128 and res == comm.size * (1 + 2j)


def test_any_all(comm):
    pnp = Reduction(comm)
    only_rank0 = np.array([comm.rank == 0])
    assert pnp.any(only_rank0)
    assert pnp.all(only_rank0) == (comm.size == 1)
    assert pnp.any(np.array([], dtype=bool)) == np.False_


def test_parallel_assert(comm):
    parallel_assert(comm, True)
    with pytest.raises(AssertionError, match="rank 0 only"):
        parallel_assert(comm, comm.rank != 0, "rank 0 only")


def test_assert_all_allclose(comm):
    assert_all_allclose(comm, np.ones(3), np.ones(3))
    with pytest.raises(AssertionError):
        # fails on the last rank only, raises on every rank
        assert_all_allclose(comm, np.ones(3),
                            np.ones(3) * (2 if comm.rank == comm.size - 1 else 1))


def test_same_reduction_group_mugrid():
    world = _mugrid_world()
    assert same_reduction_group(Reduction(world), Reduction(world))
    assert same_reduction_group(Reduction(world), Reduction(world.dup()))
    assert same_reduction_group(Reduction(world), np) == (world.size == 1)
    if world.size > 1:
        half = world.split(world.rank % 2)
        assert not same_reduction_group(Reduction(world), Reduction(half))
        # a single-rank half reduces like numpy
        assert same_reduction_group(Reduction(half), np) == (half.size == 1)


def test_bounded_lbfgs_with_mugrid_comm():
    """The distributed optimizer runs on a protocol communicator: minimise
    sum_i (x_i - c_i)^2 over all ranks, with bounds."""
    world = _mugrid_world()
    pnp = Reduction(world)
    c = np.linspace(-1.0, 2.0, 5) + world.rank

    def fun(x):
        return pnp.sum((x - c) ** 2), 2 * (x - c)

    # Interior start: from x0 = 0 (on the lower bound) the serial optimizer
    # stalls on rank 0's data too, with or without a communicator.
    res = l_bfgs_bounded(fun, np.full_like(c, 0.75), jac=True, bounds_lo=0.0,
                         bounds_hi=1.5, gtol=1e-6, comm=world)
    assert res.success
    np.testing.assert_allclose(res.x, np.clip(c, 0.0, 1.5), atol=1e-6)
