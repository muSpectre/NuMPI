#
# Copyright 2026 Lars Pastewka
#
# ### MIT license
#

import numpy as np
import pytest

from NuMPI.Optimization import l_bfgs_bounded
from NuMPI.Testing.Assertions import assert_all_allclose, parallel_assert
from NuMPI.Tools import Reduction

# ----------------------------------------------------------------------------
# Analytic ground truth: on the quadratic 0.5 ||x - y||^2 with lo <= x <= hi,
# the minimiser is the Euclidean projection clip(y, lo, hi).
# ----------------------------------------------------------------------------


def _quad(y):
    def fun_grad(x):
        return 0.5 * float(np.sum((x - y) ** 2)), (x - y)

    return fun_grad


def _quad_weighted(y, w):
    """Ill-conditioned quadratic 0.5 * sum(w_i (x_i - y_i)^2)."""
    def fun_grad(x):
        d = x - y
        return 0.5 * float(np.sum(w * d * d)), w * d

    return fun_grad


def test_unbounded_recovers_analytic_solution():
    """
    No bounds at all: the minimum of 0.5 ||x - y||^2 is y; the iteration
    should hit it to full precision.
    """
    rng = np.random.default_rng(0)
    N = 64
    y = rng.normal(size=N)

    res = l_bfgs_bounded(_quad(y), np.zeros(N), jac=True, gtol=1e-10)

    assert res.success, res.message
    np.testing.assert_allclose(res.x, y, atol=1e-8)


def test_box_matches_clip():
    """
    Box only, scalar bounds: minimiser is np.clip(y, lo, hi).
    """
    rng = np.random.default_rng(1)
    N = 128
    y = rng.normal(size=N) * 0.6 + 0.3
    lo, hi = 0.0, 1.0

    x_true = np.clip(y, lo, hi)
    res = l_bfgs_bounded(
        _quad(y), np.full(N, 0.5), jac=True,
        bounds_lo=lo, bounds_hi=hi, gtol=1e-10,
    )

    assert res.success, res.message
    np.testing.assert_allclose(res.x, x_true, atol=1e-8)
    assert res.x.min() >= lo - 1e-12
    assert res.x.max() <= hi + 1e-12


def test_box_with_array_bounds():
    """
    Per-index bounds (array lo/hi). Must still clip to y.
    """
    rng = np.random.default_rng(2)
    N = 256
    y = rng.normal(size=N)
    lo = rng.uniform(-1.0, -0.1, size=N)
    hi = rng.uniform(0.1, 1.0, size=N)

    x_true = np.clip(y, lo, hi)
    res = l_bfgs_bounded(
        _quad(y), np.zeros(N), jac=True,
        bounds_lo=lo, bounds_hi=hi, gtol=1e-10,
    )

    assert res.success, res.message
    np.testing.assert_allclose(res.x, x_true, atol=1e-8)


def test_one_sided_bound():
    """
    Only an upper bound (lower unbounded): clip from above only.
    """
    rng = np.random.default_rng(3)
    N = 96
    y = rng.normal(size=N) + 0.5
    hi = 0.25

    x_true = np.minimum(y, hi)
    res = l_bfgs_bounded(
        _quad(y), np.zeros(N), jac=True, bounds_hi=hi, gtol=1e-10,
    )

    assert res.success, res.message
    np.testing.assert_allclose(res.x, x_true, atol=1e-8)


def test_zero_mask_pins_contact_nodes():
    """
    zero_mask indices stay at zero; the rest solves the box QP.
    """
    rng = np.random.default_rng(4)
    N = 64
    y = rng.normal(size=N) * 0.3 + 0.5
    zero_mask = np.zeros(N, dtype=bool)
    zero_mask[::8] = True

    res = l_bfgs_bounded(
        _quad(y), np.full(N, 0.5), jac=True,
        bounds_lo=0.0, bounds_hi=1.0, zero_mask=zero_mask, gtol=1e-10,
    )

    assert res.success, res.message
    assert np.all(res.x[zero_mask] == 0.0)
    # Free indices match clipped y.
    free = ~zero_mask
    np.testing.assert_allclose(res.x[free], np.clip(y[free], 0.0, 1.0), atol=1e-8)


def test_initial_guess_outside_bounds_is_projected():
    """
    Infeasible x0 must be silently projected before the loop starts.
    """
    rng = np.random.default_rng(5)
    N = 32
    y = rng.normal(size=N)
    lo, hi = -2.0, 2.0
    x0 = np.full(N, 10.0)  # grossly outside

    x_true = np.clip(y, lo, hi)
    res = l_bfgs_bounded(
        _quad(y), x0, jac=True,
        bounds_lo=lo, bounds_hi=hi, gtol=1e-10,
    )

    assert res.success, res.message
    np.testing.assert_allclose(res.x, x_true, atol=1e-8)


def test_separate_jac_callable():
    """
    Passing ``jac`` as a callable (rather than ``True``) exercises the
    non-fused code path.
    """
    rng = np.random.default_rng(6)
    N = 32
    y = rng.normal(size=N)

    def fun(x):
        return 0.5 * float(np.sum((x - y) ** 2))

    def jac(x):
        return x - y

    res = l_bfgs_bounded(
        fun, np.zeros(N), jac=jac,
        bounds_lo=-0.5, bounds_hi=0.5, gtol=1e-10,
    )

    assert res.success, res.message
    np.testing.assert_allclose(res.x, np.clip(y, -0.5, 0.5), atol=1e-8)


# ----------------------------------------------------------------------------
# MPI-aware tests: per-rank slices of the global arrays, globally reduced
# scalar energy, cross-rank solution agreement.
# ----------------------------------------------------------------------------


def _distribute(global_arr, comm):
    """Slice the global array into the local rank's chunk (even split)."""
    size = comm.Get_size()
    rank = comm.Get_rank()
    n = len(global_arr)
    step = n // size
    if rank == size - 1:
        return global_arr[rank * step:].copy()
    return global_arr[rank * step:(rank + 1) * step].copy()


def test_mpi_unbounded(comm):
    """
    Distributed quadratic, no bounds. Each rank checks its local slice
    against the analytic minimiser (also sliced).
    """
    pnp = Reduction(comm)
    N_global = 256
    rng = np.random.default_rng(0)
    y_global = rng.normal(size=N_global)

    y_local = _distribute(y_global, comm)
    x_true_local = y_local.copy()

    def fun_grad(x):
        diff = x - y_local
        return 0.5 * pnp.sum(diff * diff).item(), diff

    res = l_bfgs_bounded(
        fun_grad, np.zeros_like(y_local), jac=True, comm=comm, gtol=1e-10,
    )

    parallel_assert(comm, res.success, res.message)
    assert_all_allclose(comm, res.x, x_true_local, atol=1e-8)


def test_mpi_box(comm):
    """
    Distributed box-constrained quadratic. Oracle is ``np.clip(y, lo, hi)``
    applied locally.
    """
    pnp = Reduction(comm)
    N_global = 512
    rng = np.random.default_rng(1)
    y_global = rng.normal(size=N_global)

    y_local = _distribute(y_global, comm)
    x_true_local = np.clip(y_local, 0.0, 1.0)

    def fun_grad(x):
        diff = x - y_local
        return 0.5 * pnp.sum(diff * diff).item(), diff

    res = l_bfgs_bounded(
        fun_grad, np.full_like(y_local, 0.5), jac=True,
        bounds_lo=0.0, bounds_hi=1.0, comm=comm, gtol=1e-10,
    )

    parallel_assert(comm, res.success, res.message)
    assert_all_allclose(comm, res.x, x_true_local, atol=1e-8)
    parallel_assert(comm, res.x.min() >= -1e-12 and res.x.max() <= 1 + 1e-12)


def test_mpi_matches_serial(comm):
    """
    Run the same box QP in serial and in the comm-configured mode; verify
    the distributed solution matches the concatenated serial one.
    """
    pnp = Reduction(comm)
    N_global = 256
    rng = np.random.default_rng(42)
    y_global = rng.uniform(size=N_global)

    # Serial reference.
    def fun_grad_serial(x):
        diff = x - y_global
        return 0.5 * float(np.sum(diff * diff)), diff

    res_serial = l_bfgs_bounded(
        fun_grad_serial, np.full(N_global, 0.5), jac=True,
        bounds_lo=0.0, bounds_hi=1.0, gtol=1e-10,
    )
    assert res_serial.success, res_serial.message

    # Distributed.
    y_local = _distribute(y_global, comm)

    def fun_grad_par(x):
        diff = x - y_local
        return 0.5 * pnp.sum(diff * diff).item(), diff

    res_par = l_bfgs_bounded(
        fun_grad_par, np.full_like(y_local, 0.5), jac=True,
        bounds_lo=0.0, bounds_hi=1.0, comm=comm, gtol=1e-10,
    )
    parallel_assert(comm, res_par.success, res_par.message)

    x_serial_local = _distribute(res_serial.x, comm)
    assert_all_allclose(comm, res_par.x, x_serial_local, atol=1e-8)


# ----------------------------------------------------------------------------
# Termination criteria, fallback paths, diagnostics and input guards.
# ----------------------------------------------------------------------------


def test_already_converged_at_start():
    """
    If the starting iterate already satisfies the gradient tolerance, the
    solver returns immediately (before entering the iteration loop).
    """
    rng = np.random.default_rng(9)
    N = 32
    y = rng.normal(size=N)

    # x0 == y is the exact (unconstrained) minimiser, so the residual is zero.
    res = l_bfgs_bounded(_quad(y), y.copy(), jac=True, gtol=1e-8)
    assert res.success, res.message
    assert res.nit == 0
    assert res.message == "CONVERGENCE: NORM_OF_PROJECTED_GRADIENT_<=_GTOL"


def test_ftol_convergence():
    """gtol disabled: terminate via the relative function-reduction (ftol)."""
    rng = np.random.default_rng(10)
    N = 64
    y = rng.normal(size=N)

    res = l_bfgs_bounded(
        _quad(y), np.zeros(N), jac=True,
        gtol=0, ftol=1e-8, xtol=0,
    )
    assert res.success, res.message
    assert res.message == "CONVERGENCE: REL_REDUCTION_OF_F_<=_FTOL"


def test_xtol_convergence():
    """gtol and ftol disabled: terminate via the step-size (xtol) criterion."""
    rng = np.random.default_rng(11)
    N = 64
    y = rng.normal(size=N)

    res = l_bfgs_bounded(
        _quad(y), np.zeros(N), jac=True,
        gtol=0, ftol=0, xtol=1e-8,
    )
    assert res.success, res.message
    assert res.message == "CONVERGENCE: NORM_OF_VARIABLE_STEP_<=_XTOL"


def test_maxiter_not_converged():
    """
    All tolerances disabled with a single allowed iteration: the optimizer
    must report failure with the max-iterations message.
    """
    rng = np.random.default_rng(12)
    N = 64
    y = rng.normal(size=N)

    res = l_bfgs_bounded(
        _quad(y), np.zeros(N), jac=True,
        gtol=0, ftol=0, xtol=0, maxiter=1,
    )
    assert not res.success
    assert res.message == "NO CONVERGENCE: MAXITERATIONS REACHED"


def test_linesearch_failure_returns_unsuccessful():
    """
    ``max_halvings=0`` makes every projected Armijo search fail immediately,
    so both the L-BFGS step and the steepest-descent fallback bail out and
    the solver returns ``success == False``.
    """
    rng = np.random.default_rng(13)
    N = 32
    y = rng.normal(size=N)

    res = l_bfgs_bounded(
        _quad(y), np.zeros(N), jac=True,
        gtol=1e-10, max_halvings=0,
    )
    assert not res.success
    assert res.message == "CONVERGENCE: line-search did not converge"


def test_maxcor_history_eviction():
    """
    An ill-conditioned quadratic needs more iterations than ``maxcor``, so the
    L-BFGS history fills and the oldest (s, y) pair is evicted. The solve must
    still converge to the analytic minimum.
    """
    N = 40
    rng = np.random.default_rng(14)
    y = rng.normal(size=N)
    w = np.logspace(0, 3, N)  # condition number ~1e3

    res = l_bfgs_bounded(
        _quad_weighted(y, w), np.zeros(N), jac=True,
        maxcor=2, gtol=1e-8, maxiter=500,
    )
    assert res.success, res.message
    # More iterations than maxcor guarantees the eviction branch ran.
    assert res.nit > 2
    np.testing.assert_allclose(res.x, y, atol=1e-6)


def test_disp_prints_progress(capsys):
    """disp=True prints an iteration table to stdout."""
    rng = np.random.default_rng(15)
    N = 16
    y = rng.normal(size=N)

    res = l_bfgs_bounded(_quad(y), np.zeros(N), jac=True, gtol=1e-10, disp=True)
    assert res.success, res.message
    assert capsys.readouterr().out != ""


def test_callback_invoked():
    """callback fires at least once with the current (local) iterate."""
    rng = np.random.default_rng(16)
    N = 16
    y = rng.normal(size=N)

    seen = []
    res = l_bfgs_bounded(
        _quad(y), np.zeros(N), jac=True, gtol=1e-10,
        callback=lambda x: seen.append(np.asarray(x).copy()),
    )
    assert res.success, res.message
    assert len(seen) >= 1
    assert seen[-1].shape == (N,)


def test_jac_false_not_implemented():
    """Numerical (finite-difference) gradients are not implemented."""
    y = np.zeros(8)
    with pytest.raises(NotImplementedError):
        l_bfgs_bounded(_quad(y), np.zeros(8), jac=False)


def test_rejects_comm_and_pnp(comm):
    """Supplying both comm and pnp is ambiguous and must raise."""
    y = np.zeros(8)
    with pytest.raises(RuntimeError):
        l_bfgs_bounded(_quad(y), np.zeros(8), jac=True,
                       comm=comm, pnp=Reduction(comm))
