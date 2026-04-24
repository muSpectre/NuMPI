#
# Copyright 2026 Lars Pastewka
#
# ### MIT license
#

import numpy as np
import pytest

from NuMPI.Optimization import LinearConstraint

# ----------------------------------------------------------------------------
# Constructor and sanity checks.
# ----------------------------------------------------------------------------


def test_zero_weights_rejected():
    with pytest.raises(ValueError):
        LinearConstraint(np.zeros(5), target=1.0)


def test_size_attribute():
    c = LinearConstraint(np.array([1.0, 2.0, 3.0]), target=0.0)
    assert c.size == 3


# ----------------------------------------------------------------------------
# Lagrange multiplier and tangent projection.
# ----------------------------------------------------------------------------


def test_multiplier_pure_a_direction():
    a = np.array([1.0, 2.0, 3.0, 4.0])
    c = LinearConstraint(a, target=0.0)
    # grad = 2.5 * a must give lam = 2.5.
    assert abs(c.multiplier(2.5 * a) - 2.5) < 1e-14


def test_multiplier_additive():
    rng = np.random.default_rng(0)
    a = rng.normal(size=32) ** 2 + 0.1
    c = LinearConstraint(a, target=0.0)
    # Gradient with mixed a-direction and perp-direction components.
    perp = rng.normal(size=32)
    perp -= (np.dot(a, perp) / np.dot(a, a)) * a  # make perp orthogonal to a
    lam_true = -0.7
    g = lam_true * a + perp
    assert abs(c.multiplier(g) - lam_true) < 1e-12


def test_multiplier_mask():
    # Restricting the multiplier to a subset of indices should ignore the
    # unmasked indices entirely.
    a = np.array([1.0, 2.0, 3.0, 4.0])
    c = LinearConstraint(a, target=0.0)
    g = np.array([10.0, -20.0, 30.0, -40.0])
    mask = np.array([True, True, False, False])
    a_m, g_m = a[mask], g[mask]
    expected = np.dot(a_m, g_m) / np.dot(a_m, a_m)
    assert abs(c.multiplier(g, mask=mask) - expected) < 1e-14


def test_tangent_orthogonality():
    rng = np.random.default_rng(1)
    for N in (8, 64, 512):
        a = rng.normal(size=N) ** 2 + 0.1
        c = LinearConstraint(a, target=0.0)
        g = rng.normal(size=N)
        t = c.tangent(g)
        assert abs(np.dot(a, t)) < 1e-10


# ----------------------------------------------------------------------------
# Feasibility projection.
# ----------------------------------------------------------------------------


def test_project_already_feasible_is_identity():
    # y exactly satisfies <a, y> = target and lies in the box: projection
    # returns y unchanged (tol-exactness on the constraint).
    a = np.array([1.0, 2.0, 3.0, 4.0])
    c = LinearConstraint(a, target=5.0)
    y = np.array([0.5, 0.5, 0.5, 0.5])  # <a, y> = 5.0
    x = c.project(y, lo=0.0, hi=1.0)
    np.testing.assert_allclose(x, y, atol=1e-12)
    assert abs(np.dot(a, x) - 5.0) < 1e-12


def test_project_restores_feasibility():
    a = np.array([1.0, 2.0, 3.0, 4.0])
    c = LinearConstraint(a, target=4.0)
    # y violates bounds and constraint.
    y = np.array([2.0, -1.0, 0.5, 0.8])
    x = c.project(y, lo=0.0, hi=1.0)
    assert np.all((x >= 0.0) & (x <= 1.0))
    assert abs(np.dot(a, x) - 4.0) < 1e-12


def test_project_matches_scipy_qp():
    # For a linear equality + box, the projection is the unique minimizer
    # of ||x - y||^2 over F. Verify against a direct scipy QP solver.
    scipy_optimize = pytest.importorskip("scipy.optimize")
    rng = np.random.default_rng(7)
    N = 32
    a = rng.uniform(0.1, 2.0, size=N)
    y = rng.normal(size=N)
    target = 0.5 * np.sum(a)
    c = LinearConstraint(a, target=target)

    x_ours = c.project(y, lo=0.0, hi=1.0)

    # scipy reference: direct QP
    bnds = [(0.0, 1.0)] * N
    linear = {"type": "eq", "fun": lambda x: np.dot(a, x) - target}
    res = scipy_optimize.minimize(
        lambda x: 0.5 * np.sum((x - y) ** 2),
        x0=np.full(N, 0.5),
        jac=lambda x: x - y,
        bounds=bnds,
        constraints=linear,
        method="trust-constr",
        options={"gtol": 1e-12, "xtol": 1e-12, "maxiter": 1000},
    )
    assert res.success
    np.testing.assert_allclose(x_ours, res.x, atol=1e-6)


def test_project_zero_mask_pins_contact_nodes():
    a = np.array([1.0, 2.0, 0.0, 4.0])  # a=0 at contact node
    c = LinearConstraint(a, target=3.0)
    zero_mask = np.array([False, False, True, False])
    x = c.project(np.array([0.5, 0.5, 0.9, 0.5]), lo=0.0, hi=1.0, zero_mask=zero_mask)
    assert x[2] == 0.0
    assert abs(np.dot(a, x) - 3.0) < 1e-12
    assert np.all((x >= 0.0) & (x <= 1.0))


def test_project_uniform_weight_matches_analytic_mean():
    # Uniform weights, no active bounds at optimum: projection is a uniform
    # shift of y. We pick a y whose required shift leaves every node inside
    # (0, 1) so the closed-form `x = y - mu` comparison is unambiguous.
    N = 128
    a = np.ones(N)
    target = 0.5 * N
    c = LinearConstraint(a, target=target)
    rng = np.random.default_rng(3)
    y = rng.uniform(0.3, 0.7, size=N)
    x = c.project(y, lo=0.0, hi=1.0)
    assert abs(np.mean(x) - 0.5) < 1e-12
    shift = x - y
    np.testing.assert_allclose(shift, shift[0] * np.ones(N), atol=1e-12)


def test_project_target_unreachable_raises_when_above_cap():
    # target > sum(a * hi) is infeasible.
    a = np.array([1.0, 2.0, 3.0])
    c = LinearConstraint(a, target=100.0)
    with pytest.raises(RuntimeError):
        c.project(np.array([0.5, 0.5, 0.5]), lo=0.0, hi=1.0)


def test_project_target_unreachable_raises_when_below_floor():
    a = np.array([1.0, 2.0, 3.0])
    c = LinearConstraint(a, target=-100.0)
    with pytest.raises(RuntimeError):
        c.project(np.array([0.5, 0.5, 0.5]), lo=0.0, hi=1.0)
