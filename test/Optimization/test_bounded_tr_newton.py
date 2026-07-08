#
# Copyright 2026 Lars Pastewka
#
# ### MIT license
#

import numpy as np

from NuMPI.Optimization import l_bfgs_bounded, tr_newton_bounded
from NuMPI.Testing.Assertions import assert_all_allclose, parallel_assert
from NuMPI.Tools import Reduction

# ----------------------------------------------------------------------------
# Analytic ground truths.
#
# Convex quadratic 0.5 sum(w_i (x_i - y_i)^2) on a box: minimiser is
# clip(y, lo, hi), and hessp is the diagonal map v -> w*v.
#
# Indefinite diagonal quadratic 0.5 sum(d_i x_i^2) - g_i x_i on [-1, 1]:
# for d_i > 0 the minimiser is clip(g_i/d_i, -1, 1); for d_i < 0 the
# one-dimensional problem is concave, so the minimiser sits at the box
# endpoint sign(g_i) (f(1) - f(-1) = -2 g_i). This exercises the Steihaug
# negative-curvature exit and the free-set handling together.
# ----------------------------------------------------------------------------


def _quad(y, w=1.0):
    def fun_grad(x):
        d = x - y
        return 0.5 * float(np.sum(w * d * d)), w * d

    def hessp(x, v):
        return w * v

    return fun_grad, hessp


def _indefinite(diag, g):
    def fun_grad(x):
        return float(np.sum(0.5 * diag * x * x - g * x)), diag * x - g

    def hessp(x, v):
        return diag * v

    return fun_grad, hessp


def _indefinite_minimiser(diag, g):
    x = np.where(diag > 0, np.clip(g / np.where(diag > 0, diag, 1.0), -1, 1),
                 np.sign(g))
    # d_i < 0 and g_i == 0: both endpoints tie; pick +1 (any is optimal).
    x = np.where((diag < 0) & (g == 0), 1.0, x)
    return x


def test_unbounded_quadratic():
    """Convex quadratic without bounds: exact Newton model, so the first
    interior Steihaug solve is the Newton step and convergence is immediate."""
    rng = np.random.default_rng(0)
    N = 64
    y = rng.normal(size=N)
    fun_grad, hessp = _quad(y)

    res = tr_newton_bounded(fun_grad, np.zeros(N), hessp, jac=True,
                            gtol=1e-10, delta_max=10.0)

    assert res.success, res.message
    np.testing.assert_allclose(res.x, y, atol=1e-8)
    # Quadratic + exact Hessian: a handful of outer iterations at most (the
    # radius may need to grow from delta0 first).
    assert res.nit <= 10


def test_box_matches_clip():
    """Box-constrained ill-conditioned quadratic: minimiser is clip(y)."""
    rng = np.random.default_rng(1)
    N = 128
    y = rng.normal(size=N) * 0.6 + 0.3
    w = np.exp(rng.uniform(0, 6, size=N))  # condition number ~ e^6
    lo, hi = 0.0, 1.0
    fun_grad, hessp = _quad(y, w)

    res = tr_newton_bounded(fun_grad, np.full(N, 0.5), hessp, jac=True,
                            bounds_lo=lo, bounds_hi=hi, gtol=1e-10)

    assert res.success, res.message
    np.testing.assert_allclose(res.x, np.clip(y, lo, hi), atol=1e-8)
    assert res.x.min() >= lo - 1e-12
    assert res.x.max() <= hi + 1e-12


def test_negative_curvature():
    """Indefinite quadratic on [-1, 1]: concave directions must be driven to
    the box boundary through the Steihaug negative-curvature exit."""
    rng = np.random.default_rng(2)
    N = 96
    diag = rng.uniform(-1.0, 2.0, size=N)
    diag[np.abs(diag) < 0.1] = 0.5  # keep away from singular
    g = rng.normal(size=N)
    fun_grad, hessp = _indefinite(diag, g)

    res = tr_newton_bounded(fun_grad, np.zeros(N), hessp, jac=True,
                            bounds_lo=-1.0, bounds_hi=1.0, gtol=1e-8,
                            maxiter=500)

    assert res.success, res.message
    np.testing.assert_allclose(res.x, _indefinite_minimiser(diag, g),
                               atol=1e-6)


def test_agrees_with_lbfgs():
    """Same box quadratic through both bounded optimizers -> same minimiser
    (they share feasible set and convergence measure). The quadratic is
    ill-conditioned, so L-BFGS gets a tolerance it can actually meet; the
    Newton model lets the trust region converge much tighter."""
    rng = np.random.default_rng(3)
    N = 64
    y = rng.normal(size=N) * 0.7
    w = np.exp(rng.uniform(0, 2, size=N))
    fun_grad, hessp = _quad(y, w)
    x_true = np.clip(y, -0.5, 0.5)

    res_tr = tr_newton_bounded(fun_grad, np.zeros(N), hessp, jac=True,
                               bounds_lo=-0.5, bounds_hi=0.5, gtol=1e-10)
    res_lb = l_bfgs_bounded(fun_grad, np.zeros(N), jac=True,
                            bounds_lo=-0.5, bounds_hi=0.5, gtol=1e-5)

    assert res_tr.success, res_tr.message
    assert res_lb.success, res_lb.message
    np.testing.assert_allclose(res_tr.x, x_true, atol=1e-8)
    np.testing.assert_allclose(res_lb.x, x_true, atol=1e-4)
    np.testing.assert_allclose(res_tr.x, res_lb.x, atol=1e-4)


def test_zero_mask():
    """Pinned indices stay exactly zero."""
    rng = np.random.default_rng(4)
    N = 64
    y = rng.normal(size=N)
    mask = np.zeros(N, dtype=bool)
    mask[::3] = True
    fun_grad, hessp = _quad(y)

    res = tr_newton_bounded(fun_grad, np.zeros(N), hessp, jac=True,
                            zero_mask=mask, gtol=1e-10, delta_max=10.0)

    assert res.success, res.message
    np.testing.assert_allclose(res.x[mask], 0.0, atol=0.0)
    np.testing.assert_allclose(res.x[~mask], y[~mask], atol=1e-8)


def test_nd_x0():
    """n-D x0 round-trips: iterate/gradient keep the caller's shape."""
    rng = np.random.default_rng(5)
    shape = (8, 8)
    y = rng.normal(size=shape)
    fun_grad, hessp = _quad(y)

    res = tr_newton_bounded(fun_grad, np.zeros(shape), hessp, jac=True,
                            gtol=1e-10, delta_max=10.0)

    assert res.success, res.message
    assert res.x.shape == shape
    assert res.jac.shape == shape
    np.testing.assert_allclose(res.x, y, atol=1e-8)


def test_accuracy_hooks_rescue_noisy_objective():
    """Synthetic 'truncated inner solve': the objective carries a
    deterministic O(noise_amp) perturbation, `fun_error` reports the bound
    and `request_accuracy` tightens it. The hooks must (a) be exercised and
    (b) let the run converge to the true minimiser despite an initial noise
    level far above gtol."""
    rng = np.random.default_rng(6)
    N = 64
    y = rng.normal(size=N) * 0.4
    w = np.exp(rng.uniform(0, 2, size=N))

    state = {"amp": 1e-1, "requests": 0}

    def fun_grad(x):
        d = x - y
        f = 0.5 * float(np.sum(w * d * d))
        # Deterministic pseudo-noise, O(amp) in f and in the gradient --
        # mimics the error of a truncated state solve.
        f_noise = state["amp"] * np.sin(1e3 * float(np.sum(x)))
        g_noise = state["amp"] * np.sin(1e3 * x + 1.0)
        return f + f_noise, w * d + g_noise

    def hessp(x, v):
        return w * v

    def fun_error():
        return state["amp"]

    def request_accuracy(target):
        state["requests"] += 1
        state["amp"] = min(state["amp"], 0.5 * target)

    res = tr_newton_bounded(
        fun_grad, np.zeros(N), hessp, jac=True,
        bounds_lo=-1.0, bounds_hi=1.0, gtol=1e-6, maxiter=500,
        fun_error=fun_error, request_accuracy=request_accuracy,
    )

    assert state["requests"] > 0  # the control loop actually fired
    assert res.success, res.message
    x_true = np.clip(y, -1.0, 1.0)
    np.testing.assert_allclose(res.x, x_true, atol=1e-4)


def test_callback_and_histories():
    """Callback fires per accepted iterate; diagnostics are recorded."""
    rng = np.random.default_rng(7)
    N = 32
    y = rng.normal(size=N)
    fun_grad, hessp = _quad(y)

    iterates = []
    res = tr_newton_bounded(fun_grad, np.zeros(N), hessp, jac=True,
                            gtol=1e-10, delta_max=10.0,
                            callback=lambda x: iterates.append(x.copy()))

    assert res.success
    assert len(iterates) >= 1
    assert len(res.delta_history) == len(res.rho_history)
    assert res.nb_hessp > 0


# ----------------------------------------------------------------------------
# MPI-aware tests: per-rank slices, globally reduced scalar f.
# ----------------------------------------------------------------------------


def _distribute(global_arr, comm):
    size = comm.Get_size()
    rank = comm.Get_rank()
    return np.array_split(global_arr, size)[rank]


def test_mpi_box(comm):
    """Distributed box-constrained quadratic; oracle is the local slice of
    clip(y, lo, hi)."""
    pnp = Reduction(comm)
    N_global = 256
    rng = np.random.default_rng(0)
    y_global = rng.normal(size=N_global) * 0.6 + 0.3
    w_global = np.exp(rng.uniform(0, 4, size=N_global))
    lo, hi = 0.0, 1.0

    y = _distribute(y_global, comm)
    w = _distribute(w_global, comm)

    def fun_grad(x):
        d = x - y
        return pnp.sum(0.5 * w * d * d).item(), w * d

    def hessp(x, v):
        return w * v

    res = tr_newton_bounded(fun_grad, np.full_like(y, 0.5), hessp, jac=True,
                            bounds_lo=lo, bounds_hi=hi, comm=comm,
                            gtol=1e-10)

    parallel_assert(comm, res.success, res.message)
    assert_all_allclose(comm, res.x, np.clip(y, lo, hi), atol=1e-8)


def test_mpi_negative_curvature(comm):
    """Distributed indefinite quadratic on [-1, 1]."""
    pnp = Reduction(comm)
    N_global = 192
    rng = np.random.default_rng(1)
    diag_global = rng.uniform(-1.0, 2.0, size=N_global)
    diag_global[np.abs(diag_global) < 0.1] = 0.5
    g_global = rng.normal(size=N_global)

    diag = _distribute(diag_global, comm)
    g = _distribute(g_global, comm)

    def fun_grad(x):
        return pnp.sum(0.5 * diag * x * x - g * x).item(), diag * x - g

    def hessp(x, v):
        return diag * v

    res = tr_newton_bounded(fun_grad, np.zeros_like(g), hessp, jac=True,
                            bounds_lo=-1.0, bounds_hi=1.0, comm=comm,
                            gtol=1e-8, maxiter=500)

    parallel_assert(comm, res.success, res.message)
    assert_all_allclose(comm, res.x, _indefinite_minimiser(diag, g),
                        atol=1e-6)
