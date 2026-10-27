import pytest

from .. import MPI

#
# This is adopted from runtests:
# https://github.com/bccp/runtests
# BSD-2-Clause License
#
# `MPI` is mpi4py's or, if mpi4py is not installed, NuMPI's stub; with the
# stub, the world has a single process and only fixtures of size one run.
#

communicators = {}


class WorldTooSmall(Exception):
    pass


def create_comm(size):
    if MPI.COMM_WORLD.size < size:
        raise WorldTooSmall

    color = 0 if MPI.COMM_WORLD.rank < size else 1
    if size not in communicators:
        if MPI.COMM_WORLD.size == size:
            comm = MPI.COMM_WORLD
        elif size == 1:
            comm = MPI.COMM_SELF
        else:
            comm = MPI.COMM_WORLD.Split(color)
        communicators[size] = comm

    return communicators[size], color


def MPITestFixture(commsize, scope="function"):
    """Create a test fixture for MPI Communicators of various commsizes"""

    @pytest.fixture(params=commsize, scope=scope)
    def fixture(request):
        MPI.COMM_WORLD.barrier()
        try:
            comm, color = create_comm(request.param)

            if color != 0:
                pytest.skip("Not using communicator %d" % (request.param))
                return None
            else:
                return comm

        except WorldTooSmall:
            pytest.skip("Not using communicator %d" % request.param)
            return None

    return fixture


serial_only = pytest.mark.skipif(
    MPI.COMM_WORLD.Get_size() > 1,
    reason="Test runs on a single process only; run it with pytest instead of mpirun")
"""Marker for tests that must not run under MPI with multiple processes"""
