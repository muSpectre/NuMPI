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
Tests for the pickle-based (lowercase) object communication methods of the
MPI stub. These mimic mpi4py's interface for arbitrary Python objects and
reduce to trivial operations in serial.
"""

import pytest

from NuMPI import MPIStub


@pytest.fixture
def comm():
    return MPIStub.Intracomm()


def test_rank_and_size(comm):
    assert comm.Get_rank() == 0
    assert comm.Get_size() == 1
    assert comm.rank == 0
    assert comm.size == 1


def test_gather(comm):
    obj = {"nested": [1, 2, 3]}
    assert comm.gather(obj) == [obj]
    assert comm.gather(obj, root=0) == [obj]
    with pytest.raises(ValueError):
        comm.gather(obj, root=1)


def test_allgather(comm):
    obj = ("a", 1)
    assert comm.allgather(obj) == [obj]


def test_bcast(comm):
    obj = {"key": "value"}
    assert comm.bcast(obj) is obj
    with pytest.raises(ValueError):
        comm.bcast(obj, root=1)


def test_scatter(comm):
    assert comm.scatter(["only"]) == "only"
    with pytest.raises(ValueError):
        comm.scatter(["only"], root=1)
    # Sequence length must match communicator size
    with pytest.raises(ValueError):
        comm.scatter(["one", "two"])


def test_reduce(comm):
    assert comm.reduce(5) == 5
    assert comm.reduce(5, op=MPIStub.MAX) == 5
    with pytest.raises(ValueError):
        comm.reduce(5, root=1)


def test_allreduce(comm):
    assert comm.allreduce(7) == 7
    assert comm.allreduce(7, op=MPIStub.MIN) == 7
