#
# Copyright 2026 Lars Pastewka
#
# ### MIT license
#

"""
Tests for the Polonsky-Keer constrained conjugate gradient (with restart on
active-set changes). Mirrors the structure of ``test_cg.py`` so the two CG
variants are exercised on the same distributed problems.
"""

import numpy as np
import pytest

from NuMPI.Testing.Assertions import assert_all_allclose, parallel_assert
from NuMPI.Tools import Reduction
from test.Optimization.MPIMinimizationProblems import MPI_Quadratic

from NuMPI.Optimization.CCGWithRestart import \
    constrained_conjugate_gradients_with_restart as ccg_restart

try:
    import scipy.optimize
    _scipy_present = True
except ModuleNotFoundError:
    _scipy_present = False


def test_exported_from_package():
    """The solver is reachable from the package namespace."""
    from NuMPI.Optimization import \
        constrained_conjugate_gradients_with_restart  # noqa: F401


def test_ccg_restart(comm):
    n = 128
    obj = MPI_Quadratic(n, pnp=Reduction(comm))
    xstart = np.random.normal(size=obj.nb_subdomain_grid_pts)

    res = ccg_restart(
        obj.f_grad, obj.hessian_product, args=(2,),
        x0=xstart, communicator=comm,
    )
    parallel_assert(comm, res.success, res.message)


def test_ccg_restart_active_bounds(comm):
    n = 128
    np.random.seed(0)
    obj = MPI_Quadratic(n, pnp=Reduction(comm), xmin=np.random.normal(size=n))
    xstart = np.random.normal(size=obj.nb_subdomain_grid_pts)

    res = ccg_restart(
        obj.f_grad, obj.hessian_product, args=(2,),
        x0=xstart, communicator=comm,
    )
    parallel_assert(comm, res.success, res.message)
    # Solution must respect the (default, zero) lower bound.
    parallel_assert(comm, (res.x >= -1e-12).all())


@pytest.mark.skipif(not _scipy_present, reason='scipy not present')
def test_ccg_restart_arbitrary_bounds(comm):
    """Bound-constrained minimiser must match scipy's L-BFGS-B (serial)."""
    n = 128
    np.random.seed(0)
    obj = MPI_Quadratic(n, pnp=Reduction(comm))
    xstart = np.random.normal(size=obj.nb_subdomain_grid_pts)
    bounds = np.random.normal(size=obj.nb_subdomain_grid_pts)

    res = ccg_restart(
        obj.f_grad, obj.hessian_product, args=(2,),
        x0=xstart, communicator=comm, bounds=bounds, gtol=1e-8,
    )
    parallel_assert(comm, res.success, res.message)
    parallel_assert(comm, (res.x >= bounds - 1e-12).all())

    if comm.size == 1:
        bnds = tuple([(b, None) for b in bounds])
        res_ref = scipy.optimize.minimize(
            obj.f_grad, args=(2,), x0=xstart, bounds=bnds,
            method="L-BFGS-B", jac=True, options=dict(gtol=1e-8, ftol=0))
        assert res_ref.success, res_ref.message
        assert_all_allclose(comm, res.x, res_ref.x, atol=1e-6)


def test_ccg_restart_unbounded_recovers_minimum(comm):
    """
    With all components unbounded (bounds = -inf) the restart variant reduces
    to ordinary CG and must recover the analytic minimum of the quadratic.
    """
    np.random.seed(1)
    n = 64
    xstart = np.random.normal(size=n)
    target = np.random.normal(size=n)
    curvature = 3.0

    if comm is not None:
        rank, size = comm.Get_rank(), comm.Get_size()
        step = n // size
        sub = slice(rank * step, None) if rank == size - 1 \
            else slice(rank * step, (rank + 1) * step)
        xstart, target = xstart[sub], target[sub]

    def f_grad(x, scale):
        diff = x - target
        return 0.5 * scale * np.dot(diff, diff), scale * diff

    def hessp(d, scale):
        return scale * d

    res = ccg_restart(
        f_grad, hessp, x0=xstart, args=(curvature,), communicator=comm,
        bounds=np.full_like(xstart, -np.inf), gtol=1e-10,
    )
    parallel_assert(comm, res.success, res.message)
    assert_all_allclose(comm, res.x, target, atol=1e-7)


def test_ccg_restart_mean_val(comm):
    n = 128
    obj = MPI_Quadratic(n, pnp=Reduction(comm))
    xstart = np.random.normal(size=obj.nb_subdomain_grid_pts)

    res = ccg_restart(
        obj.f_grad, obj.hessian_product, args=(2,),
        x0=xstart, mean_val=1.0, communicator=comm,
    )
    parallel_assert(comm, res.success, res.message)
    # The prescribed global mean must hold at the solution.
    pnp = Reduction(comm)
    mean_x = pnp.sum(res.x) / pnp.sum(res.x.size)
    parallel_assert(comm, abs(float(mean_x) - 1.0) < 1e-6)


def test_ccg_restart_jac_false(comm):
    """Gradient-only objective via jac=False."""
    np.random.seed(3)
    n = 64
    obj = MPI_Quadratic(n, pnp=Reduction(comm))
    xstart = np.random.normal(size=obj.nb_subdomain_grid_pts)

    res = ccg_restart(
        obj.grad, obj.hessian_product, args=(2,),
        jac=False, x0=xstart, communicator=comm,
    )
    parallel_assert(comm, res.success, res.message)


def test_ccg_restart_split_jac(comm):
    """Separate objective and gradient callables (jac=callable)."""
    np.random.seed(4)
    n = 64
    obj = MPI_Quadratic(n, pnp=Reduction(comm))
    xstart = np.random.normal(size=obj.nb_subdomain_grid_pts)

    res = ccg_restart(
        obj.f, obj.hessian_product, args=(2,),
        jac=obj.grad, x0=xstart, communicator=comm,
    )
    parallel_assert(comm, res.success, res.message)


def test_ccg_restart_callback_invoked(comm):
    n = 64
    obj = MPI_Quadratic(n, pnp=Reduction(comm))
    xstart = np.random.normal(size=obj.nb_subdomain_grid_pts)

    seen = []
    res = ccg_restart(
        obj.f_grad, obj.hessian_product, args=(2,),
        x0=xstart, communicator=comm, callback=lambda x: seen.append(1),
    )
    parallel_assert(comm, res.success, res.message)
    parallel_assert(comm, len(seen) >= 1)


def test_ccg_restart_maxiter_not_converged(comm):
    n = 128
    np.random.seed(0)
    obj = MPI_Quadratic(n, pnp=Reduction(comm), xmin=np.random.normal(size=n))
    xstart = np.random.normal(size=obj.nb_subdomain_grid_pts)

    res = ccg_restart(
        obj.f_grad, obj.hessian_product, args=(2,),
        x0=xstart, communicator=comm, maxiter=1, gtol=1e-14,
    )
    parallel_assert(comm, not res.success)
    parallel_assert(comm, res.message == "NO CONVERGENCE: MAXITERATIONS REACHED")


def test_ccg_restart_rejects_none_x0():
    with pytest.raises(ValueError):
        ccg_restart(lambda x: (0.0, x), lambda d: d, x0=None)


def test_ccg_restart_rejects_invalid_jac():
    with pytest.raises(ValueError):
        ccg_restart(lambda x: (0.0, x), lambda d: d,
                    x0=np.ones(4), jac="nope")


def test_ccg_restart_rejects_invalid_mean_val():
    with pytest.raises(ValueError):
        ccg_restart(lambda x: (0.0, x), lambda d: d,
                    x0=np.ones(4), mean_val="nope")


def test_ccg_restart_rejects_invalid_residual_plot():
    with pytest.raises(ValueError):
        ccg_restart(lambda x: (0.0, x), lambda d: d,
                    x0=np.ones(4), residual_plot="nope")
