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

from NuMPI.Optimization import BilinearConstraint, LinearConstraint, QuadraticConstraint, l_bfgs_qc_projected
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
        # A far trial point of a line search may overflow; the infinite energy
        # makes the solver reject it.
        with np.errstate(over="ignore", invalid="ignore"):
            energy = (
                0.25 * (g - p) ** 4 + 0.5 * (h - q) ** 2 + np.cosh(0.3 * g * h)
            )
            coupling = 0.3 * np.sinh(0.3 * g * h)
            grad = np.concatenate(
                [(g - p) ** 3 + coupling * h, (h - q) + coupling * g])
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


# Badly scaled problems: g -> factor * g, h -> h / factor leaves sum g h, the
# minimum and the multiplier unchanged, but multiplies the curvature of the
# objective along g by factor^-2 and along h by factor^2, and the condition
# number of its Hessian by up to factor^4.
FACTORS = [10.0, 1000.0, 1e-3]
BOXES = {"free": (None, None), "box": (0.0, 2.0)}


def _badly_scaled(comm, kind, factor, box):
    """
    The reference problem and its badly scaled twin.

    Returns:
        ``(reference, twin, scale, bounds)``. ``reference()`` solves the
        well-scaled problem; ``twin(scale=None)`` the same problem in the
        variables ``scale * (g, h)``, where ``scale`` holds ``factor`` for g
        and ``1 / factor`` for h; ``bounds`` are the bounds of the twin.
    """
    pnp = Reduction(comm)
    nodes = _slice(comm)
    lo, hi = BOXES[box]
    fun_grad, x0, k = _objective(nodes, pnp)
    scale = np.concatenate([np.full(k, factor), np.full(k, 1.0 / factor)])
    fun_grad_twin, _, _ = _objective(nodes, pnp, scale=scale)
    bounds = (None if lo is None else lo * scale, None if hi is None else hi * scale)

    def reference():
        return l_bfgs_qc_projected(
            fun_grad, x0, _constraint(kind, k, 6.0, pnp), bounds_lo=lo,
            bounds_hi=hi, comm=comm, gtol=1e-7)

    def twin(**kwargs):
        return l_bfgs_qc_projected(
            fun_grad_twin, x0 * scale, _constraint(kind, k, 6.0, pnp),
            bounds_lo=bounds[0], bounds_hi=bounds[1], comm=comm, gtol=1e-7,
            **kwargs)

    return reference, twin, scale, bounds


# The scale that undoes the factor exactly, and the one the solver finds for
# the two fields of a BilinearConstraint by itself.
RULES = [("quadratic", "given"), ("bilinear", "given"), ("bilinear", "auto")]


@pytest.mark.parametrize("kind, rule", RULES, ids=["-".join(r) for r in RULES])
@pytest.mark.parametrize("box", BOXES)
@pytest.mark.parametrize("factor", FACTORS)
def test_scale_solves_a_badly_scaled_problem_like_the_well_scaled_one(
        comm, kind, rule, box, factor):
    # With the given scale the solver iterates on the variables of the
    # reference problem, so it takes the same path up to rounding. The
    # automatic scale balances the curvatures of the objective along g and h,
    # which are of the same order in the reference problem.
    reference, twin, scale, (lo, hi) = _badly_scaled(comm, kind, factor, box)
    pnp = Reduction(comm)

    ref = reference()
    res = twin(scale=scale if rule == "given" else "auto")

    parallel_assert(comm, ref.success, ref.message)
    parallel_assert(comm, res.success, res.message)
    # The automatic scale starts from another feasible point and iterates on
    # other variables than the reference, so its path differs.
    most = ref.nit + 10 if rule == "given" else 2 * ref.nit + 10
    parallel_assert(comm, res.nit <= most, f"{res.nit} vs {ref.nit} iterations")
    parallel_assert(comm, res.constr_violation < 1e-11, str(res.constr_violation))
    parallel_assert(comm, abs(res.fun - ref.fun) < 1e-9 * abs(ref.fun))
    if rule == "given":
        parallel_assert(comm, abs(res.max_grad - ref.max_grad) < 1e-7)
        parallel_assert(comm, np.array_equal(res.scale, scale))
    else:
        # sigma on g and 1 / sigma on h, with sigma of the order of the factor.
        sigma, inverse = np.split(res.scale, 2)
        parallel_assert(comm, np.all(sigma == sigma[:1]) and np.all(inverse == inverse[:1]))
        parallel_assert(comm, np.allclose(sigma * inverse, 1.0, rtol=1e-14))
        parallel_assert(
            comm, np.all((sigma > factor / 3.0) & (sigma < factor * 3.0)), str(sigma[:1]))
    parallel_assert(comm, np.allclose(res.multiplier, ref.multiplier, atol=1e-5))
    # x and jac come back in the variables of the twin. The minimum has soft
    # directions, so the points agree less closely than the energies.
    assert_all_allclose(comm, res.x / scale, ref.x, atol=1e-3)
    assert_all_allclose(comm, res.jac * scale, ref.jac, atol=1e-3)
    energy, grad = _objective(_slice(comm), pnp, scale=scale)[0](res.x)
    parallel_assert(comm, res.fun == energy)
    assert_all_allclose(comm, res.jac, grad, atol=0.0)
    if lo is not None:
        parallel_assert(
            comm, np.all(res.x >= lo) and np.all(res.x <= hi * (1.0 + 1e-14)))


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("box", BOXES)
@pytest.mark.parametrize("factor", FACTORS)
def test_badly_scaled_problem_is_out_of_reach_without_scale(comm, kind, box, factor):
    # Documents the sensitivity that `scale` removes: within five times the
    # iterations of the reference, the solver has not converged.
    reference, twin, _, _ = _badly_scaled(comm, kind, factor, box)

    ref = reference()
    res = twin(maxiter=5 * ref.nit)

    parallel_assert(comm, ref.success, ref.message)
    parallel_assert(comm, not res.success, f"converged in {res.nit} iterations")
    # What it returns is feasible all the same.
    parallel_assert(comm, res.constr_violation < 1e-11, str(res.constr_violation))


def test_automatic_scale_needs_the_two_fields_of_a_bilinear_constraint(comm):
    pnp = Reduction(comm)
    fun_grad, x0, k = _objective(_slice(comm), pnp)

    with pytest.raises(ValueError, match="BilinearConstraint"):
        l_bfgs_qc_projected(
            fun_grad, x0, _constraint("quadratic", k, 6.0, pnp), scale="auto",
            comm=comm)
    with pytest.raises(ValueError, match="unknown scale"):
        l_bfgs_qc_projected(
            fun_grad, x0, _constraint("bilinear", k, 6.0, pnp), scale="automatic",
            comm=comm)


def test_automatic_scale_probes_the_objective_inside_the_bounds(comm):
    # An objective that is not defined outside the box, started with entries
    # on both bounds.
    pnp = Reduction(comm)
    fun_grad, x0, k = _objective(_slice(comm), pnp)
    lo, hi = 0.2, 1.2
    outside = [0]

    def guarded(x):
        outside[0] += np.count_nonzero((x < lo) | (x > hi))
        return fun_grad(x)

    res = l_bfgs_qc_projected(
        guarded, np.clip(x0, lo, hi), _constraint("bilinear", k, 6.0, pnp),
        bounds_lo=lo, bounds_hi=hi, scale="auto", comm=comm, gtol=1e-7)

    parallel_assert(comm, res.success, res.message)
    parallel_assert(comm, pnp.sum(np.array([outside[0]])) == 0)


def test_automatic_scale_equalises_the_norms_without_positive_curvature(comm):
    # Linear in g: the curvature along g is zero, so the scale comes from the
    # norms of g and h at the restored starting point.
    pnp = Reduction(comm)
    nodes = _slice(comm)
    k = nodes.size
    rng = np.random.default_rng(3)
    p = rng.uniform(0.5, 1.5, N_NODES)[nodes]
    q = rng.uniform(0.5, 1.5, N_NODES)[nodes]
    x0 = np.concatenate([np.full(k, 40.0), np.full(k, 0.1)])
    lo = 0.0
    hi = np.concatenate([np.full(k, 80.0), np.full(k, 2.0)])

    def fun_grad(x):
        g, h = np.split(x, 2)
        energy = p * g + 0.5 * (h - q) ** 2
        return pnp.sum(energy).item(), np.concatenate([p, h - q])

    res = l_bfgs_qc_projected(
        fun_grad, x0, _constraint("bilinear", k, 6.0, pnp), bounds_lo=lo,
        bounds_hi=hi, scale="auto", comm=comm, gtol=1e-7)

    g, h = np.split(BilinearConstraint(k, 6.0, pnp=pnp).project(x0, lo=lo, hi=hi), 2)
    sigma = (pnp.sum(g * g) / pnp.sum(h * h)) ** 0.25
    parallel_assert(comm, res.success, res.message)
    parallel_assert(comm, np.allclose(res.scale[:k], sigma) and sigma > 5.0, str(sigma))
    parallel_assert(comm, np.allclose(res.scale[k:], 1.0 / sigma))


@pytest.mark.parametrize("kind", KINDS)
def test_scalar_scale_and_bad_scale(comm, kind):
    # A scalar scale applies to every unknown; it does not change the balance
    # of g and h, and the result is that of the unscaled run.
    pnp = Reduction(comm)
    fun_grad, x0, k = _objective(_slice(comm), pnp)
    kwargs = dict(bounds_lo=0.0, bounds_hi=2.0, comm=comm, gtol=1e-7)

    ref = l_bfgs_qc_projected(fun_grad, x0, _constraint(kind, k, 6.0, pnp), **kwargs)
    res = l_bfgs_qc_projected(
        fun_grad, x0, _constraint(kind, k, 6.0, pnp), scale=1.0, **kwargs)

    parallel_assert(comm, res.success and res.nit == ref.nit, res.message)
    assert_all_allclose(comm, res.x, ref.x, atol=1e-12)

    for bad in (0.0, -1.0, np.inf, np.nan):
        with pytest.raises(ValueError, match="scale"):
            l_bfgs_qc_projected(
                fun_grad, x0, _constraint(kind, k, 6.0, pnp), scale=bad, **kwargs)


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
