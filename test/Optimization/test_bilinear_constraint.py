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
The bilinear constraint sum_i g_i h_i = V of two fields x = (g, h), as a
``QuadraticConstraint`` with a matrix-free product and as a
``BilinearConstraint``.

Every rank holds a slice of the nodes; its local unknowns are its entries of g
followed by its entries of h.
"""

import numpy as np
import pytest

from NuMPI.Optimization import (BilinearConstraint, LinearConstraint,
                                QuadraticConstraint, l_bfgs_qc_projected)
from NuMPI.Testing.Assertions import assert_all_allclose, parallel_assert
from NuMPI.Tools import Reduction

N_NODES = 24
KINDS = ["quadratic", "bilinear"]


def _slice(comm):
    """Indices of the nodes of this rank."""
    return np.array_split(np.arange(N_NODES), comm.Get_size())[comm.Get_rank()]


def _local(g, h, nodes):
    return np.concatenate([g[nodes], h[nodes]])


def _swap(u):
    k = u.size // 2
    return np.concatenate([u[k:], u[:k]])


def _constraint(kind, k, target, pnp):
    if kind == "quadratic":
        return QuadraticConstraint(np.zeros(2 * k), _swap, target, pnp=pnp)
    return BilinearConstraint(k, target, pnp=pnp)


def _objective(nodes, pnp, scale=1.0):
    """
    f = sum_i (g_i - p_i)^4 / 4 + (h_i - q_i)^2 / 2 + cosh(0.3 g_i h_i) on the
    given nodes, as a function of scale * (g, h). Not convex, and not
    quadratic.
    """
    rng = np.random.default_rng(7)
    p = rng.normal(size=N_NODES)[nodes] + 1.0
    q = rng.normal(size=N_NODES)[nodes] + 1.0
    k = nodes.size

    def fun_grad(xs):
        g, h = np.split(xs / scale, 2)
        energy = (
            0.25 * (g - p) ** 4 + 0.5 * (h - q) ** 2 + np.cosh(0.3 * g * h)
        )
        coupling = 0.3 * np.sinh(0.3 * g * h)
        grad = np.concatenate([(g - p) ** 3 + coupling * h, (h - q) + coupling * g])
        return pnp.sum(energy).item(), grad / scale

    return fun_grad, np.concatenate([p, q]), k


def _kkt_violation(res, lo, hi, pnp):
    """
    Largest violation of the first-order optimality conditions: the gradient
    of the Lagrangian vanishes inside the bounds, is not negative on the lower
    bound and not positive on the upper bound.
    """
    r = res.jac - res.multiplier[0] * _swap(res.x)
    at_lo = np.zeros(r.shape, dtype=bool) if lo is None else res.x <= lo
    at_hi = np.zeros(r.shape, dtype=bool) if hi is None else res.x >= hi
    violation = np.where(at_lo, np.minimum(r, 0.0), np.where(at_hi, np.maximum(r, 0.0), r))
    return pnp.max(np.abs(violation), initial=0.0)


CASES = {
    "positive_target": (6.0, None, None),
    "negative_target": (-3.0, None, None),
    "loose_box": (6.0, 0.0, 2.0),
    "tight_box": (6.0, 0.2, 0.8),
}


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("case", CASES)
def test_solver_finds_a_stationary_point_of_a_nonlinear_objective(comm, kind, case):
    target, lo, hi = CASES[case]
    pnp = Reduction(comm)
    nodes = _slice(comm)
    fun_grad, x0, k = _objective(nodes, pnp)

    res = l_bfgs_qc_projected(
        fun_grad, x0, _constraint(kind, k, target, pnp),
        bounds_lo=lo, bounds_hi=hi, comm=comm, gtol=1e-7,
    )

    parallel_assert(comm, res.success, res.message)
    parallel_assert(comm, res.constr_violation < 1e-11, str(res.constr_violation))
    parallel_assert(comm, _kkt_violation(res, lo, hi, pnp) < 1e-6)
    if lo is not None:
        parallel_assert(comm, np.all(res.x >= lo) and np.all(res.x <= hi))

    # The same problem on one process, on the global arrays.
    all_nodes = np.arange(N_NODES)
    fun_grad_serial, x0_serial, _ = _objective(all_nodes, np)
    ref = l_bfgs_qc_projected(
        fun_grad_serial, x0_serial, _constraint(kind, N_NODES, target, np),
        bounds_lo=lo, bounds_hi=hi, gtol=1e-7,
    )
    g_ref, h_ref = np.split(ref.x, 2)
    assert_all_allclose(comm, res.x, _local(g_ref, h_ref, nodes), atol=1e-5)
    parallel_assert(comm, np.allclose(res.multiplier, ref.multiplier, atol=1e-5))


def _random_point(comm, seed, low, high):
    rng = np.random.default_rng(seed)
    g, h = rng.uniform(low, high, (2, N_NODES))
    nodes = _slice(comm)
    return _local(g, h, nodes), nodes.size


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("seed", range(5))
def test_restored_point_is_feasible(comm, kind, seed):
    pnp = Reduction(comm)
    y, k = _random_point(comm, seed, -0.5, 2.5)
    qc = _constraint(kind, k, 10.0, pnp)

    x = qc.project(y, lo=0.0, hi=2.0)

    parallel_assert(comm, x is not None)
    parallel_assert(comm, abs(qc.value(x)[0] - 10.0) < 1e-11)
    parallel_assert(comm, np.all(x >= 0.0) and np.all(x <= 2.0))


@pytest.mark.parametrize("kind", KINDS)
def test_restoration_leaves_a_feasible_point_and_fixed_entries_alone(comm, kind):
    pnp = Reduction(comm)
    y, k = _random_point(comm, 0, 0.0, 2.0)
    qc = _constraint(kind, k, 10.0, pnp)
    x = qc.project(y, lo=0.0, hi=2.0)

    assert_all_allclose(comm, qc.project(x, lo=0.0, hi=2.0), x, atol=1e-12)

    fixed = np.arange(2 * k) % 3 == 0
    x_fixed = qc.project(y, lo=0.0, hi=2.0, fixed=fixed)
    parallel_assert(comm, x_fixed is not None)
    parallel_assert(comm, abs(qc.value(x_fixed)[0] - 10.0) < 1e-11)
    parallel_assert(comm, np.array_equal(x_fixed[fixed], y[fixed]))


def test_bilinear_projection_is_the_nearest_feasible_point(comm):
    # Optimality of min |x - y|^2 / 2 on the feasible set: with the multiplier
    # mu, the vector x - y + mu B x vanishes inside the bounds and points into
    # the box on them.
    pnp = Reduction(comm)
    for seed in range(5):
        y, k = _random_point(comm, seed, -0.5, 2.5)
        qc = BilinearConstraint(k, 10.0, pnp=pnp)

        x = qc.project(y, lo=0.0, hi=2.0)

        normal = _swap(x)
        inside = (x > 0.0) & (x < 2.0)
        mu = -pnp.sum(np.where(inside, normal * (x - y), 0.0)) / pnp.sum(
            np.where(inside, normal * normal, 0.0))
        r = x - y + mu * normal
        violation = np.where(
            inside, r, np.where(x <= 0.0, np.minimum(r, 0.0), np.maximum(r, 0.0)))
        parallel_assert(comm, pnp.max(np.abs(violation), initial=0.0) < 1e-9)
        parallel_assert(comm, abs(mu) < 1.0)

        # The restoration along the clipped normal path is feasible, too, and
        # cannot be nearer.
        x_path = QuadraticConstraint(np.zeros(2 * k), _swap, 10.0, pnp=pnp).project(
            y, lo=0.0, hi=2.0)
        parallel_assert(
            comm,
            pnp.sum((x - y) ** 2) <= pnp.sum((x_path - y) ** 2) * (1.0 + 1e-12))


def test_restoration_agrees_with_the_projection_to_second_order(comm):
    # Without bounds, the restoration along the normals at y and the Euclidean
    # projection differ by the square of the distance of y from the feasible
    # set: a quarter of the distance gives a sixteenth of the difference.
    pnp = Reduction(comm)
    start, k = _random_point(comm, 1, 0.5, 1.5)
    direction, _ = _random_point(comm, 2, -1.0, 1.0)
    exact = BilinearConstraint(k, 20.0, pnp=pnp)
    path = QuadraticConstraint(np.zeros(2 * k), _swap, 20.0, pnp=pnp)
    x = exact.project(start)
    d = path.tangent(direction, x)
    d /= np.sqrt(pnp.sum(d * d))

    difference = []
    for alpha in (0.4, 0.2):
        y = x + alpha * d
        delta = path.project(y) - exact.project(y)
        difference.append(np.sqrt(pnp.sum(delta * delta)))

    parallel_assert(comm, 0.0 < difference[1] < difference[0] / 10.0, str(difference))


@pytest.mark.parametrize("kind", KINDS)
def test_point_across_the_cone_is_not_restored(comm, kind):
    # g = 1, h = -1 on every node: sum g h < 0, and along the normal (h, g)
    # the sum never becomes positive.
    pnp = Reduction(comm)
    k = _slice(comm).size
    y = np.concatenate([np.ones(k), -np.ones(k)])

    parallel_assert(comm, _constraint(kind, k, float(N_NODES), pnp).project(y) is None)


def test_restoration_onto_a_linear_constraint_is_the_euclidean_projection(comm):
    pnp = Reduction(comm)
    nodes = _slice(comm)
    for seed in range(20):
        rng = np.random.default_rng(seed)
        a = rng.uniform(0.1, 2.0, N_NODES)[nodes]
        y = 2.0 * rng.normal(size=N_NODES)[nodes]
        target = rng.uniform(1.0, 5.0)

        x = QuadraticConstraint(a, None, target, pnp=pnp).project(y, lo=0.0, hi=1.5)

        parallel_assert(comm, x is not None, f"seed {seed}")
        assert_all_allclose(
            comm, x, LinearConstraint(a, target, pnp=pnp).project(y, lo=0.0, hi=1.5),
            atol=1e-10)


@pytest.mark.parametrize("kind", KINDS)
def test_vanishing_normal_ends_the_run(comm, kind):
    # sum g h = 0 is a cone; at its tip g = h = 0 the normal (h, g) vanishes.
    pnp = Reduction(comm)
    fun_grad, _, k = _objective(_slice(comm), pnp)

    res = l_bfgs_qc_projected(
        fun_grad, np.zeros(2 * k), _constraint(kind, k, 0.0, pnp), comm=comm)

    parallel_assert(comm, not res.success)
    parallel_assert(comm, "SINGULAR" in res.message, res.message)


@pytest.mark.parametrize("kind", KINDS)
def test_scale_restores_the_convergence_of_a_badly_scaled_problem(comm, kind):
    # g -> 100 g, h -> h / 100 leaves sum g h and the minimum unchanged.
    pnp = Reduction(comm)
    nodes = _slice(comm)
    fun_grad, x0, k = _objective(nodes, pnp)
    scale = np.concatenate([np.full(k, 100.0), np.full(k, 0.01)])
    fun_grad_scaled, _, _ = _objective(nodes, pnp, scale=scale)
    kwargs = dict(bounds_lo=0.0, comm=comm, gtol=1e-7, maxiter=150)

    ref = l_bfgs_qc_projected(fun_grad, x0, _constraint(kind, k, 6.0, pnp), **kwargs)
    scaled = l_bfgs_qc_projected(
        fun_grad_scaled, x0 * scale, _constraint(kind, k, 6.0, pnp), scale=scale,
        **kwargs)
    unscaled = l_bfgs_qc_projected(
        fun_grad_scaled, x0 * scale, _constraint(kind, k, 6.0, pnp), **kwargs)

    parallel_assert(comm, ref.success and scaled.success, scaled.message)
    parallel_assert(comm, scaled.nit <= ref.nit + 5, f"{scaled.nit} {ref.nit}")
    # gtol bounds scale * gradient, which is 100 times looser for h.
    assert_all_allclose(comm, scaled.x / scale, ref.x, atol=1e-3)
    assert_all_allclose(comm, scaled.jac * scale, ref.jac, atol=1e-3)
    parallel_assert(comm, np.allclose(scaled.multiplier, ref.multiplier, atol=1e-4))
    parallel_assert(comm, scaled.constr_violation < 1e-11)
    # Without `scale` the same problem is out of reach.
    parallel_assert(
        comm, not unscaled.success or unscaled.nit > 3 * ref.nit, str(unscaled.nit))


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("tolerance", ["xtol", "ftol"])
def test_tolerance_exit_returns_the_accepted_iterate(comm, kind, tolerance):
    pnp = Reduction(comm)
    fun_grad, x0, k = _objective(_slice(comm), pnp)

    res = l_bfgs_qc_projected(
        fun_grad, x0, _constraint(kind, k, 6.0, pnp), bounds_lo=0.0, comm=comm,
        gtol=0.0, **{tolerance: 1e-4})

    energy, grad = fun_grad(res.x)
    parallel_assert(comm, res.success and tolerance.upper() in res.message, res.message)
    parallel_assert(comm, res.fun == energy)
    assert_all_allclose(comm, res.jac, grad, atol=0.0)


def test_products_are_reused_along_the_line_search(comm):
    # One product for the search direction and one per restored trial point;
    # the products at the iterate follow from these by linearity.
    pnp = Reduction(comm)
    fun_grad, x0, k = _objective(_slice(comm), pnp)
    calls = [0]
    evaluations = [0]

    def hessp(u):
        calls[0] += 1
        return _swap(u)

    def counted(x):
        evaluations[0] += 1
        return fun_grad(x)

    res = l_bfgs_qc_projected(
        counted, x0, QuadraticConstraint(np.zeros(2 * k), hessp, 6.0, pnp=pnp),
        comm=comm, gtol=1e-7)

    parallel_assert(comm, res.success, res.message)
    parallel_assert(
        comm, calls[0] <= evaluations[0] + res.nit + res.nit // 20 + 6,
        f"{calls[0]} products, {evaluations[0]} evaluations, {res.nit} iterations")
