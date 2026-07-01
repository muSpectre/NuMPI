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
import pytest

from NuMPI.Optimization import l_bfgs
from NuMPI.Testing.Assertions import parallel_assert
from NuMPI.Tools import Reduction

try:
    import scipy.optimize

    _scipy_present = True
except ModuleNotFoundError:
    _scipy_present = False

import test.Optimization.MinimizationProblems as mp
import test.Optimization.MPIMinimizationProblems as mpi_mp


@pytest.fixture
def pnp(comm):
    return Reduction(comm)


@pytest.mark.skipif(not _scipy_present, reason="scipy not present")
@pytest.mark.parametrize("Objectiveclass", [mpi_mp.MPI_Quadratic])
def test_minimize_call_column(pnp, Objectiveclass):
    """
    Checks the compatibility with a scipy.optimize.minimze call. Note that one
    should not use this type of call when running in parallel

    Parameters
    ----------
    pnp
    Objectiveclass

    Returns
    -------

    """

    comm = pnp.comm
    n = 10 + 2 * comm.Get_size()
    Objective = Objectiveclass(n, pnp=pnp)
    result = scipy.optimize.minimize(
        Objective.f_grad,
        Objective.startpoint(),
        jac=True,
        method=l_bfgs,
        args=(2,),
        options={"gtol": 1e-8, "ftol": 1e-20, "pnp": pnp},
    )
    parallel_assert(comm, result.success)
    parallel_assert(
        comm,
        pnp.max(abs(np.reshape(result.x, (-1,)) - np.reshape(Objective.xmin(), (-1,))))
        / pnp.max(abs(Objective.startpoint() - Objective.xmin()))
        < 1e-5,
    )


@pytest.mark.skipif(not _scipy_present, reason="scipy not present")
@pytest.mark.parametrize("Objectiveclass", [mpi_mp.MPI_Quadratic])
def test_minimize_call_row(pnp, Objectiveclass):
    comm = pnp.comm
    n = 10 + 2 * comm.Get_size()
    Objective = Objectiveclass(n, pnp=pnp)
    result = scipy.optimize.minimize(
        Objective.f_grad,
        Objective.startpoint().reshape(-1),
        method=l_bfgs,
        jac=True,
        args=(2,),
        options={"gtol": 1e-8, "ftol": 1e-20, "pnp": pnp},
    )
    parallel_assert(comm, result.success, "")
    parallel_assert(
        comm,
        (
            pnp.max(
                abs(np.reshape(result.x, (-1,)) - np.reshape(Objective.xmin(), (-1,)))
            )
            / pnp.max(abs(Objective.startpoint() - Objective.xmin()))
            < 1e-5
        ),
    )


@pytest.mark.parametrize("shape", [(10,), (10, 1), (2, 5), (4, 5)])
def test_shape_unchanged(shape):
    """
    This test is only serial. Includes genuine multi-axis (n-D) shapes: the
    unconstrained solver normalises the iterate to a column vector internally,
    so history and directions stay consistent regardless of x0's shape.
    """
    size = np.prod(shape)
    Objective = mp.Extended_Rosenbrock

    x0 = Objective.startpoint(size).reshape(shape)

    result = l_bfgs(
        Objective.f_grad, x0, jac=True, args=(2,), gtol=1e-10, ftol=1e-40
    )

    assert result.success, ""
    np.testing.assert_allclose(
        np.reshape(result.x, (-1,)),
        np.reshape(Objective.xmin(size), (-1,)),
        rtol=1e-5,
    )

    assert result.x.shape == shape, "shape of result mismatch shape of startpoint"
    assert result.jac.shape == shape, "shape of result jac mismatch shape of startpoint"
