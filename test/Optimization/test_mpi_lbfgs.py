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


import pytest

import numpy as np
from NuMPI.Tools import Reduction
from test.Optimization.MPIMinimizationProblems import (
    MPI_Objective_Interface, MPI_Quadratic
)

import time
import test.Optimization.MinimizationProblems as mp

from NuMPI.Optimization.LBFGS import l_bfgs


def timer(fun, *args, **kwargs):
    start = time.perf_counter()
    res = fun(*args, **kwargs)
    delay = time.perf_counter() - start
    return res, delay


def test_linesearch():
    pass


@pytest.mark.parametrize("n", [10, 20, 50])
def test_quadratic_analytical_min(comm, n):
    """
    Compares the result with the analyticaly known posistion of the minimum
    :return:
    """

    def printMPI(msg):
        for i in range(comm.Get_size()):
            comm.barrier()
            if comm.Get_rank() == i:
                print("Proc {}: {}".format(i, msg))

    PObjective = MPI_Quadratic(nb_domain_grid_pts=n, pnp=Reduction(comm))

    x0 = PObjective.startpoint()

    res = l_bfgs(
        PObjective.f_grad,
        x0,
        args=(2,),
        jac=True,
        maxcor=5,
        maxiter=100,
        gtol=1e-14,
        ftol=0,  # only terminates if gradient condition is satisfied
        pnp=Reduction(comm))
    assert res.success
    assert res.message == "CONVERGENCE: NORM_OF_GRADIENT_<=_GTOL"
    np.testing.assert_allclose(res.x, PObjective.xmin(), atol=1e-13)

    assert np.abs(res.fun - PObjective.minVal(n)) < 1e-7


@pytest.mark.parametrize("n", [10, 20, 50])
def test_rosenbrock_analytical_min(comm, n):
    """
    Compares the result with the analyticaly known posistion of the minimum
    :return:
    """

    def printMPI(msg):
        for i in range(comm.Get_size()):
            comm.barrier()
            if comm.Get_rank() == i:
                print("Proc {}: {}".format(i, msg))

    Objective = mp.Extended_Rosenbrock

    PObjective = MPI_Objective_Interface(Objective, nb_domain_grid_pts=n,
                                         comm=comm)

    x0 = PObjective.startpoint()

    res = l_bfgs(
        PObjective.f_grad,
        x0,
        args=(2,),
        jac=True,
        maxcor=5,
        maxiter=100,
        gtol=1e-12,  # only terminates if gradient condition is
        ftol=0,
        pnp=Reduction(comm))
    assert res.success
    assert res.message == "CONVERGENCE: NORM_OF_GRADIENT_<=_GTOL"
    np.testing.assert_allclose(res.x, PObjective.xmin(), atol=1e-16, rtol=1e-7)

    assert np.abs(res.fun - Objective.minVal(n)) < 1e-7


def test_ftol_convergence(comm):
    """
    With ``gtol`` disabled the run must terminate via the function-value
    (``ftol``) criterion and report the matching message.
    """
    PObjective = MPI_Quadratic(nb_domain_grid_pts=20, pnp=Reduction(comm))
    res = l_bfgs(
        PObjective.f_grad, PObjective.startpoint(), args=(2,), jac=True,
        gtol=0, ftol=1e-8, xtol=0, maxiter=1000, comm=comm)
    assert res.success
    assert res.message == "CONVERGENCE: REL_REDUCTION_OF_F_<=_FTOL"


def test_xtol_convergence(comm):
    """
    With ``gtol`` and ``ftol`` disabled the run must terminate via the
    step-size (``xtol``) criterion.
    """
    PObjective = MPI_Quadratic(nb_domain_grid_pts=20, pnp=Reduction(comm))
    res = l_bfgs(
        PObjective.f_grad, PObjective.startpoint(), args=(2,), jac=True,
        gtol=0, ftol=0, xtol=1e-8, maxiter=1000, comm=comm)
    assert res.success
    assert res.message == "CONVERGENCE: NORM_OF_VARIABLE_STEP_<=_XTOL"


def test_maxiter_not_converged(comm):
    """
    All tolerances effectively disabled with too few iterations: the
    optimizer must give up and report ``success == False``.
    """
    PObjective = MPI_Objective_Interface(
        mp.Extended_Rosenbrock, nb_domain_grid_pts=20, comm=comm)
    res = l_bfgs(
        PObjective.f_grad, PObjective.startpoint(), args=(2,), jac=True,
        gtol=1e-30, ftol=0, xtol=0, maxiter=2, comm=comm)
    assert not res.success


def test_disp_and_callback(comm, capsys):
    """
    ``disp`` prints a progress table (rank 0 only); ``callback`` fires at
    least once during the iteration.
    """
    PObjective = MPI_Quadratic(nb_domain_grid_pts=20, pnp=Reduction(comm))
    seen = []
    res = l_bfgs(
        PObjective.f_grad, PObjective.startpoint(), args=(2,), jac=True,
        gtol=1e-10, comm=comm, disp=1, store_iterates=True,
        callback=lambda x: seen.append(1))
    assert res.success
    assert len(seen) >= 1
    # store_iterates=True records one OptimizeResult per iteration.
    assert len(res.iterates) >= 1
    captured = capsys.readouterr()
    if comm.rank == 0:
        assert captured.out != ""
    else:
        assert captured.out == ""


def test_rejects_comm_and_pnp(comm):
    """Supplying both ``comm`` and ``pnp`` is ambiguous and must raise."""
    PObjective = MPI_Quadratic(nb_domain_grid_pts=10, pnp=Reduction(comm))
    with pytest.raises(RuntimeError):
        l_bfgs(PObjective.f_grad, PObjective.startpoint(), args=(2,),
               jac=True, comm=comm, pnp=Reduction(comm))


def test_jac_false_not_implemented(comm):
    """Numerical (finite-difference) gradients are not implemented."""
    PObjective = MPI_Quadratic(nb_domain_grid_pts=10, pnp=Reduction(comm))
    with pytest.raises(NotImplementedError):
        l_bfgs(PObjective.f_grad, PObjective.startpoint(), args=(2,),
               jac=False, comm=comm)


@pytest.mark.skip(reason="just plotting")
def test_time_complexity(comm):
    maxcor = 5
    Objective = mp.Extended_Rosenbrock
    n = np.array([10, 100, 1000, 1e4, 1e5, 1e6], dtype=int)
    t = np.zeros(len(n), dtype=float)
    res = [None] * len(n)
    pnp = Reduction(comm)
    for i in range(len(n)):
        PObjective = MPI_Objective_Interface(Objective,
                                             nb_domain_grid_pts=n[i],
                                             comm=comm)
        x0 = PObjective.startpoint()

        res[i], t[i] = timer(l_bfgs, PObjective.f_grad, x0, jac=True,
                             maxcor=maxcor, maxiter=100000, gtol=(1e-5),
                             pnp=pnp)

        assert res[i].success

    if True:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        ax.plot(n, t / n, '+-', label="time / DOF")
        ax.plot(n, [t[i] / n[i] / res[i].nit for i in range(len(n))], '+-',
                label="time per DOF per iteration")
        ax2 = plt.twinx(ax)
        ax2.plot(n, [res[i].nit for i in range(len(n))], 'o', label="nit")
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_ylabel("t/n (s)")
        ax.set_xlabel("DOF")
        ax.legend()
        ax2.legend()
        # ax.plot(n,n,c='gray')
        plt.show(block=True)
