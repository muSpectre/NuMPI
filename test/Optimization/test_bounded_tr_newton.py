#
# Copyright 2026 Lars Pastewka
#
# ### MIT license
#

import numpy as np
import pytest

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


def _flat_quadratic(lam):
    """Very flat convex quadratic 0.5 lam |x|^2, paired with a *zero* Hessian
    model.

    The zero model has no curvature for Steihaug to work against, so every
    subproblem exits through the negative-curvature branch with a step that
    runs all the way to the trust-region boundary -- however small the
    gradient. Pinning delta at delta_max then keeps every accepted step
    boundary-limited for the whole run.
    """
    def fun_grad(x):
        return 0.5 * lam * float(x @ x), lam * x

    def hessp(x, v):
        return np.zeros_like(v)

    return fun_grad, hessp


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


def _ill_conditioned_quadratic(scale, N=64):
    """``scale * 0.5 sum(w_i (x_i - y_i)^2)`` with spread curvatures, so the
    Steihaug CG needs several iterations and its tolerance matters."""
    rng = np.random.default_rng(3)
    w = np.logspace(0, 3, N)
    y = rng.normal(size=N)
    hessps = []

    def fun_grad(x):
        d = x - y
        return scale * 0.5 * float(np.sum(w * d * d)), scale * w * d

    def hessp(x, v, tol=None):
        hessps.append(tol)
        return scale * w * v

    return fun_grad, hessp, hessps, y


def test_forcing_does_not_depend_on_objective_units():
    """Scaling the objective (and so gradient and Hessian) by a constant must
    not change the iterates: the default Steihaug tolerance is relative to
    the first projected gradient. On ||g|| itself, a 1e-6 scale demanded a
    near-exact subproblem from the first iteration and took more products.
    The scale is a power of two, so that the scaled arithmetic is exact."""
    results = []
    for scale in (1.0, 2.0**-20):
        fun_grad, hessp, hessps, y = _ill_conditioned_quadratic(scale)
        res = tr_newton_bounded(fun_grad, np.zeros_like(y), hessp, jac=True,
                                gtol=1e-8 * scale, delta_max=100.0)
        assert res.success, res.message
        np.testing.assert_allclose(res.x, y, atol=1e-6)
        results.append((res.nit, res.nb_hessp, res.x))
    (nit_a, nh_a, x_a), (nit_b, nh_b, x_b) = results
    assert (nit_a, nh_a) == (nit_b, nh_b)
    np.testing.assert_allclose(x_a, x_b, rtol=1e-10, atol=1e-12)


def test_hessp_tol_passes_the_steihaug_tolerance():
    fun_grad, hessp, hessps, y = _ill_conditioned_quadratic(1.0)
    res = tr_newton_bounded(fun_grad, np.zeros_like(y), hessp, jac=True,
                            gtol=1e-8, delta_max=100.0, hessp_tol=True)
    assert res.success, res.message
    assert len(hessps) == res.nb_hessp
    # The first subproblem is solved to 0.5, then the forcing tightens
    assert hessps[0] == 0.5
    assert all(0.0 <= t <= 0.5 for t in hessps)
    assert min(hessps) < 0.5
    # Without the option, hessp is called without a tolerance
    fun_grad, hessp, hessps, y = _ill_conditioned_quadratic(1.0)
    tr_newton_bounded(fun_grad, np.zeros_like(y), hessp, jac=True,
                      gtol=1e-8, delta_max=100.0)
    assert set(hessps) == {None}


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


def test_accuracy_hooks_stop_retrying_at_floor():
    """A caller that cannot improve its accuracy (e.g. the attainable
    residual of a single-precision solve) must not be asked over and over:
    after one fruitless request per iteration the driver proceeds with the
    best available accuracy instead of burning max_accuracy_retries
    re-evaluation rounds."""
    rng = np.random.default_rng(8)
    N = 32
    y = rng.normal(size=N) * 0.3

    state = {"requests": 0, "evals": 0}
    floor = 1e-3  # irreducible noise level

    def fun_grad(x):
        state["evals"] += 1
        d = x - y
        f = 0.5 * float(np.sum(d * d))
        return f + floor * np.sin(1e3 * float(np.sum(x))), d

    def hessp(x, v):
        return v

    def fun_error():
        return floor  # never improves

    def request_accuracy(target):
        state["requests"] += 1  # honoured but futile

    res = tr_newton_bounded(
        fun_grad, np.zeros(N), hessp, jac=True, gtol=1e-6, maxiter=50,
        delta_max=10.0,
        fun_error=fun_error, request_accuracy=request_accuracy,
    )

    # At most one futile request per outer iteration (the stagnation break),
    # not max_accuracy_retries of them.
    assert state["requests"] <= res.nit + 1
    # And the run terminates (converged or not) rather than looping.
    assert res.nit <= 50


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


def test_converges_when_steps_stay_boundary_limited():
    """A run whose every step is trust-region-limited must still stop on the
    gradient tolerance.

    Guards the unconditional convergence test. Gating it on the previous step
    having been interior -- on the grounds that a boundary-limited step means
    the model "wanted to go further" -- fails here: with ~zero curvature along
    the gradient every step ends on the boundary no matter how small the
    gradient is, so such a condition can never be discharged. A run that
    cannot stop burns its whole iteration budget and returns nit == maxiter
    with an answer that depends on maxiter, which the assertions below rule
    out.
    """
    N, lam, delta_max, x_start = 8, 1e-6, 0.1, 2.0
    fun_grad, hessp = _flat_quadratic(lam)
    # |g|inf = lam * x crosses gtol at x = 1.2, while the minimiser (x = 0) is
    # still eight boundary-limited steps away -- so a run that fails to notice
    # convergence has plenty of budget left to burn.
    gtol = 1.25e-6
    budgets = (20, 60, 200)

    runs = [
        tr_newton_bounded(fun_grad, np.full(N, x_start), hessp, jac=True,
                          bounds_lo=-10.0, bounds_hi=10.0, gtol=gtol,
                          maxiter=maxiter, delta0=delta_max,
                          delta_max=delta_max)
        for maxiter in budgets
    ]

    for res, maxiter in zip(runs, budgets):
        assert res.success, res.message
        assert float(np.max(np.abs(res.jac))) < gtol
        # Stop when converged, not when the budget runs out.
        assert res.nit < maxiter, (
            f"maxiter={maxiter}: ran the full budget (nit={res.nit}) although "
            f"|g|inf={float(np.max(np.abs(res.jac))):.3e} < gtol={gtol:.3e}"
        )
        # Every accepted step moved exactly one trust-region radius, i.e. the
        # run really did stay boundary-limited throughout (without this the
        # test could silently stop exercising the path it is guarding).
        np.testing.assert_allclose(res.x, x_start - res.nit * delta_max,
                                   atol=1e-12)
        np.testing.assert_allclose(res.delta_history, delta_max, atol=1e-12)

    # The result must not depend on the iteration budget.
    for res in runs[1:]:
        assert res.nit == runs[0].nit
        np.testing.assert_array_equal(res.x, runs[0].x)


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


# ----------------------------------------------------------------------------
# Preconditioned Steihaug CG: `precond`, `precond_metric`, energy-error stop
# with fixed (`inner_delay`) or adaptive (`inner_tau`) delay.
#
# Ground truth: the SPD quadratic 0.5 x^T A x - b^T x with A a 1D Laplacian
# plus a widely varying diagonal (ill-conditioned, not diagonal), minimiser
# A^-1 b. The Jacobi preconditioner M = diag(A) is cheap and not exact, so CG
# needs several iterations in every trust-region step.
# ----------------------------------------------------------------------------


def _spd_system(N, seed):
    rng = np.random.default_rng(seed)
    diag = np.exp(rng.uniform(0, 6, size=N))
    A = np.diag(diag + 2.0) - np.diag(np.ones(N - 1), 1) - np.diag(np.ones(N - 1), -1)
    b = rng.normal(size=N)
    return A, b


def _spd_quadratic(A, b):
    def fun_grad(x):
        Ax = A @ x
        return 0.5 * float(x @ Ax) - float(b @ x), Ax - b

    def hessp(x, v):
        return A @ v

    return fun_grad, hessp


def test_safety_factor():
    """Hand-computed values of the adaptive-delay safety factor
    S = max_j tail_j / Delta_j, tail_j = sum_{i>=j} Delta_i."""
    from NuMPI.Optimization.BoundedTRNewtonCG import _safety_factor

    # tails 15, 7, 3, 1: ratios 15/8, 7/4, 3/2 over j = 0..2 (not the last)
    assert np.isclose(_safety_factor([8.0, 4.0, 2.0, 1.0], 0), 15 / 8)
    # tails 1e5+3, 1e5+2, 2, 1; oldest = 2: tail_0, tail_1 >= 1e4 tail_2, so
    # the max starts at j = 1 (ratio 1.00002), max at j = 2 (2 / 1). Without
    # that cut, j = 0 would give 1e5 + 3.
    assert np.isclose(_safety_factor([1.0, 1e5, 1.0, 1.0], 2), 2.0)


@pytest.mark.parametrize("delay_tau", [None, 0.25])
@pytest.mark.parametrize("use_precond", [False, True])
def test_energy_stop_reaches_tolerance(delay_tau, use_precond):
    """One inner solve with a radius that is never reached: CG stops on the
    energy-error estimate, and the true energy error of the step is below the
    requested tolerance (with a small safety margin)."""
    from NuMPI.Optimization.BoundedTRNewtonCG import _steihaug_cg

    A, b = _spd_system(200, 3)
    g = -b  # model m(s) = g^T s + 0.5 s^T A s, minimiser A^-1 b
    s_exact = np.linalg.solve(A, b)
    precond = (lambda r: r / np.diag(A)) if use_precond else None
    tol = 1e-6

    s, pred, on_boundary, status, nb_hessp = _steihaug_cg(
        lambda v: A @ v, g, 1e10, tol, 10000, np, precond=precond,
        stop="energy", delay=10, delay_tau=delay_tau)

    assert status == "tolerance" and not on_boundary
    err = s - s_exact
    assert np.sqrt(err @ A @ err) <= 10 * tol * np.sqrt(s_exact @ A @ s_exact)
    assert pred > 0


def test_energy_stop_adaptive_delay_needs_at_least_min_delay():
    """With the adaptive delay, at least `delay` CG iterations are done, even
    where a loose tolerance on a well-conditioned system would allow fewer."""
    from NuMPI.Optimization.BoundedTRNewtonCG import _steihaug_cg

    rng = np.random.default_rng(4)
    w = rng.uniform(1.0, 1.5, size=200)
    b = rng.normal(size=200)
    *_, status, nb_hessp = _steihaug_cg(lambda v: w * v, -b, 1e10, 0.5, 10000, np, stop="energy", delay=1,
                                        delay_tau=0.25)
    assert status == "tolerance" and nb_hessp < 10
    for delay in (10, 30):
        *_, status, nb_hessp = _steihaug_cg(lambda v: w * v, -b, 1e10, 0.5, 10000, np, stop="energy", delay=delay,
                                            delay_tau=0.25)
        assert status == "tolerance"
        assert nb_hessp >= delay


def test_invalid_inner_stop():
    A, b = _spd_system(16, 5)
    fun_grad, hessp = _spd_quadratic(A, b)
    with pytest.raises(ValueError):
        tr_newton_bounded(fun_grad, np.zeros(16), hessp, jac=True,
                          inner_stop="bogus")


@pytest.mark.parametrize("inner_stop", ["residual", "energy"])
@pytest.mark.parametrize("inner_tau", [None, 0.25])
def test_precond_converges(inner_stop, inner_tau):
    """Jacobi-preconditioned TR Newton reaches A^-1 b, with every inner stop
    variant, and uses fewer Hessian products than without preconditioner."""
    N = 128
    A, b = _spd_system(N, 6)
    fun_grad, hessp = _spd_quadratic(A, b)
    kwargs = dict(jac=True, gtol=1e-10, delta0=1.0, delta_max=1e3, inner_tol=1e-8,
                  inner_stop=inner_stop, inner_tau=inner_tau)

    res = tr_newton_bounded(fun_grad, np.zeros(N), hessp,
                            precond=lambda r: r / np.diag(A), **kwargs)
    res_plain = tr_newton_bounded(fun_grad, np.zeros(N), hessp, **kwargs)

    assert res.success, res.message
    np.testing.assert_allclose(res.x, np.linalg.solve(A, b), atol=1e-8)
    assert res.nb_precond > 0
    assert res.nb_hessp < res_plain.nb_hessp


def test_precond_rejects_bounds():
    A, b = _spd_system(16, 7)
    fun_grad, hessp = _spd_quadratic(A, b)
    with pytest.raises(NotImplementedError):
        tr_newton_bounded(fun_grad, np.zeros(16), hessp, jac=True,
                          bounds_lo=0.0, precond=lambda r: r / np.diag(A))


def test_precond_metric_equal_to_precond_norm():
    """precond_metric(z) = M z reproduces the default trust region (the
    M-norm) exactly: same iterates, same counts."""
    N = 128
    A, b = _spd_system(N, 8)
    fun_grad, hessp = _spd_quadratic(A, b)
    M = np.diag(A)
    kwargs = dict(jac=True, gtol=1e-10, precond=lambda r: r / M)

    res = tr_newton_bounded(fun_grad, np.zeros(N), hessp, **kwargs)
    res_metric = tr_newton_bounded(fun_grad, np.zeros(N), hessp,
                                   precond_metric=lambda z: M * z, **kwargs)

    assert res.success and res_metric.success
    np.testing.assert_allclose(res_metric.x, res.x, rtol=0, atol=1e-12)
    assert res_metric.nit == res.nit
    assert res_metric.nb_hessp == res.nb_hessp
    np.testing.assert_allclose(res_metric.delta_history, res.delta_history)


@pytest.mark.parametrize("metric_diag", ["identity", "random"])
def test_precond_metric_boundary_step(metric_diag):
    """With precond_metric(z) = N z, a step that reaches the trust-region
    boundary has ||s||_N = delta (N-norm, not the preconditioner's M-norm),
    and TR Newton with that metric still converges."""
    from NuMPI.Optimization.BoundedTRNewtonCG import _steihaug_cg

    N = 128
    A, b = _spd_system(N, 9)
    M = np.diag(A)
    rng = np.random.default_rng(10)
    Nd = np.ones(N) if metric_diag == "identity" else np.exp(rng.uniform(-2, 2, size=N))
    precond = lambda r: r / M  # noqa: E731
    metric = lambda z: Nd * z  # noqa: E731

    # N-norms of the unconstrained CG iterates s_1, s_2, ...; the radius is put
    # between a later iterate (the 3rd or later) and the largest earlier one, so that the
    # boundary is crossed only after several iterations and the crossing
    # test uses the recursively propagated N s, N d
    norms = []
    for k in range(1, 40):
        s_k = _steihaug_cg(lambda v: A @ v, -b, 1e10, 1e-14, k, np, precond=precond)[0]
        norms.append(np.sqrt(s_k @ (Nd * s_k)))
    k = next(k for k in range(2, len(norms)) if norms[k] > max(norms[:k]) * 1.001)
    delta = 0.5 * (max(norms[:k]) + norms[k])

    s, pred, on_boundary, status, nb_hessp = _steihaug_cg(
        lambda v: A @ v, -b, delta, 1e-14, 10000, np, precond=precond, metric=metric)

    assert nb_hessp == k + 1
    assert on_boundary and status == "boundary"
    np.testing.assert_allclose(np.sqrt(s @ (Nd * s)), delta, rtol=1e-10)
    assert pred > 0

    fun_grad, hessp = _spd_quadratic(A, b)
    res = tr_newton_bounded(fun_grad, np.zeros(N), hessp, jac=True, gtol=1e-10,
                            precond=precond, precond_metric=metric, inner_tau=0.25)
    assert res.success, res.message
    np.testing.assert_allclose(res.x, np.linalg.solve(A, b), atol=1e-8)


def test_mpi_precond_metric_adaptive(comm):
    """Distributed: Jacobi preconditioner, trust region in another norm,
    adaptive delay; same minimiser as the serial dense solve."""
    pnp = Reduction(comm)
    N_global = 192
    rng = np.random.default_rng(11)
    w_global = np.exp(rng.uniform(0, 6, size=N_global))
    y_global = rng.normal(size=N_global)
    n_global = np.exp(rng.uniform(-1, 1, size=N_global))

    w = _distribute(w_global, comm)
    y = _distribute(y_global, comm)
    n = _distribute(n_global, comm)

    # w_i (x_i - y_i)^2 / 2 with the preconditioner 2 / w (inexact by 2x)
    def fun_grad(x):
        d = x - y
        return pnp.sum(0.5 * w * d * d).item(), w * d

    def hessp(x, v):
        return w * v

    res = tr_newton_bounded(fun_grad, np.zeros_like(y), hessp, jac=True,
                            comm=comm, gtol=1e-10,
                            precond=lambda r: 2 * r / w,
                            precond_metric=lambda z: n * z, inner_tau=0.25)

    parallel_assert(comm, res.success, res.message)
    assert_all_allclose(comm, res.x, y, atol=1e-8)
