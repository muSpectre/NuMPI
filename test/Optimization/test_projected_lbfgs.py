#
# Copyright 2026 Lars Pastewka
#
# ### MIT license
#

import numpy as np
import pytest

from NuMPI.Optimization import LinearConstraint, l_bfgs_projected
from NuMPI.Testing.Assertions import assert_all_allclose, parallel_assert
from NuMPI.Tools import Reduction

# ----------------------------------------------------------------------------
# Analytic ground truth for a quadratic with linear equality + box bounds:
# min ½ ||x - y||^2  s.t.  <a, x> = V*, lo <= x <= hi
# The optimum is the Euclidean projection of y onto F, which we also have
# in LinearConstraint.project — so we use it as the reference oracle.
# ----------------------------------------------------------------------------


def _quad(y):
    def fun_grad(x):
        return 0.5 * float(np.sum((x - y) ** 2)), (x - y)

    return fun_grad


def test_pure_equality_recovers_analytic_solution():
    """
    No bounds: the unique KKT point is x* = y - lam * a with
    lam = (<a, y> - V*)/<a, a>. L-BFGS on the quadratic must hit it exactly
    (the problem is one-step solvable: identity initial Hessian + one line
    search lands on the minimum).
    """
    rng = np.random.default_rng(0)
    N = 64
    a = np.abs(rng.normal(size=N)) + 0.1
    y = rng.normal(size=N)
    V = 2.0
    lam = (np.dot(a, y) - V) / np.dot(a, a)
    x_true = y - lam * a

    c = LinearConstraint(a, V)
    res = l_bfgs_projected(_quad(y), np.zeros(N), c, jac=True, gtol=1e-10)

    assert res.success, res.message
    np.testing.assert_allclose(res.x, x_true, atol=1e-8)
    assert abs(np.dot(a, res.x) - V) < 1e-10


def test_box_plus_uniform_mean_matches_projection():
    """
    Uniform weights, box active at some indices: result must match the
    LinearConstraint.project applied to y (they're the same QP).
    """
    rng = np.random.default_rng(1)
    N = 128
    a = np.ones(N) / N
    target = 0.3
    y = rng.normal(size=N) * 0.3 + 0.5
    c = LinearConstraint(a, target)

    x_true = c.project(y, lo=0.0, hi=1.0)
    res = l_bfgs_projected(
        _quad(y), np.full(N, 0.5), c, jac=True, bounds_lo=0.0, bounds_hi=1.0, gtol=1e-10
    )

    assert res.success, res.message
    np.testing.assert_allclose(res.x, x_true, atol=1e-8)
    assert abs(np.mean(res.x) - target) < 1e-10
    assert res.x.min() >= -1e-12 and res.x.max() <= 1 + 1e-12


def test_box_plus_non_uniform_weights():
    """
    Generalised weighted equality (the feature that motivated the refactor).
    Verify the L-BFGS result against the closed-form projection oracle.
    """
    rng = np.random.default_rng(2)
    N = 256
    a = np.abs(rng.normal(size=N)) + 0.1
    target = 0.5 * np.sum(a)
    y = rng.uniform(0.0, 1.0, size=N)
    c = LinearConstraint(a, target)

    x_true = c.project(y, lo=0.0, hi=1.0)
    res = l_bfgs_projected(
        _quad(y), np.full(N, 0.5), c, jac=True, bounds_lo=0.0, bounds_hi=1.0, gtol=1e-10
    )

    assert res.success, res.message
    np.testing.assert_allclose(res.x, x_true, atol=1e-8)
    assert abs(np.dot(a, res.x) - target) < 1e-8


def test_zero_mask_pins_contact_nodes():
    """
    Indices in zero_mask must stay at zero throughout; the constraint is
    enforced on the remaining indices only (here a_i is zero at masked
    nodes, as is the typical volume-weight case).
    """
    rng = np.random.default_rng(3)
    N = 64
    a = np.abs(rng.normal(size=N)) + 0.1
    zero_mask = np.zeros(N, dtype=bool)
    zero_mask[::8] = True  # 8 pinned nodes
    a[zero_mask] = 0.0  # zero weight at contact
    y = rng.normal(size=N) * 0.3 + 0.5
    target = 0.3 * np.sum(a)
    c = LinearConstraint(a, target)

    res = l_bfgs_projected(
        _quad(y),
        np.full(N, 0.5),
        c,
        jac=True,
        bounds_lo=0.0,
        bounds_hi=1.0,
        zero_mask=zero_mask,
        gtol=1e-10,
    )

    assert res.success, res.message
    assert np.all(res.x[zero_mask] == 0.0)
    assert abs(np.dot(a, res.x) - target) < 1e-8


def test_rejects_missing_constraint():
    rng = np.random.default_rng(4)
    y = rng.normal(size=8)
    with pytest.raises(ValueError):
        l_bfgs_projected(_quad(y), np.zeros(8), None, jac=True)


def test_initial_guess_outside_feasible_set_is_projected():
    """
    An infeasible x0 is silently projected before the loop starts — the
    final solution must still be correct.
    """
    rng = np.random.default_rng(5)
    N = 32
    a = np.abs(rng.normal(size=N)) + 0.1
    target = 1.0
    y = rng.normal(size=N)
    c = LinearConstraint(a, target)
    x0 = np.full(N, 10.0)  # grossly outside bounds AND violates equality

    x_true = c.project(y, lo=-2.0, hi=2.0)
    res = l_bfgs_projected(
        _quad(y), x0, c, jac=True, bounds_lo=-2.0, bounds_hi=2.0, gtol=1e-10
    )

    assert res.success, res.message
    np.testing.assert_allclose(res.x, x_true, atol=1e-8)


# ----------------------------------------------------------------------------
# MPI-aware tests. The distributed problem uses per-rank slices of the global
# arrays (a, y, bounds, x0), a globally-reduced scalar energy via the `comm`
# fixture's Reduction, and verifies the solution agrees across all ranks.
#
# Under serial execution (no MPI) these run on one "rank" via the MPIStub;
# under ``mpirun -n K`` pytest they exercise real cross-rank reductions.
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


def test_mpi_pure_equality(comm):
    """
    Distributed quadratic with a linear equality constraint. Each rank sees
    its own slice of a, y, x0; the solution is checked rank-locally against
    the analytic closed-form minimiser (also sliced).
    """
    pnp = Reduction(comm)
    N_global = 256
    # Build the global problem on every rank (same seed), then slice.
    rng = np.random.default_rng(0)
    a_global = np.abs(rng.normal(size=N_global)) + 0.1
    y_global = rng.normal(size=N_global)
    V = 2.0
    lam_true = (np.dot(a_global, y_global) - V) / np.dot(a_global, a_global)
    x_true_global = y_global - lam_true * a_global

    a_local = _distribute(a_global, comm)
    y_local = _distribute(y_global, comm)
    x_true_local = _distribute(x_true_global, comm)

    def fun_grad(x):
        diff = x - y_local
        return 0.5 * pnp.sum(diff * diff).item(), diff

    lc = LinearConstraint(a_local, V, pnp=pnp)
    res = l_bfgs_projected(
        fun_grad, np.zeros_like(y_local), lc, jac=True, comm=comm, gtol=1e-10
    )

    parallel_assert(comm, res.success, res.message)
    assert_all_allclose(comm, res.x, x_true_local, atol=1e-8)
    # Constraint satisfied globally.
    parallel_assert(comm, abs(pnp.sum(a_local * res.x).item() - V) < 1e-10)


def test_mpi_box_plus_linear_constraint(comm):
    """
    Distributed box + linear-equality problem. Oracle is
    LinearConstraint.project built on every rank (still a distributed
    LinearConstraint), and we compare rank-locally.
    """
    pnp = Reduction(comm)
    N_global = 512
    rng = np.random.default_rng(1)
    a_global = np.abs(rng.normal(size=N_global)) + 0.1
    y_global = rng.uniform(size=N_global)
    target = 0.5 * a_global.sum()

    a_local = _distribute(a_global, comm)
    y_local = _distribute(y_global, comm)

    lc = LinearConstraint(a_local, target, pnp=pnp)
    x_oracle_local = lc.project(y_local, lo=0.0, hi=1.0)

    def fun_grad(x):
        diff = x - y_local
        return 0.5 * pnp.sum(diff * diff).item(), diff

    res = l_bfgs_projected(
        fun_grad,
        np.full_like(y_local, 0.5),
        lc,
        jac=True,
        bounds_lo=0.0,
        bounds_hi=1.0,
        comm=comm,
        gtol=1e-10,
    )

    parallel_assert(comm, res.success, res.message)
    assert_all_allclose(comm, res.x, x_oracle_local, atol=1e-7)
    parallel_assert(comm, abs(pnp.sum(a_local * res.x).item() - target) < 1e-8)
    parallel_assert(comm, res.x.min() >= -1e-12 and res.x.max() <= 1 + 1e-12)


def test_mpi_matches_serial(comm):
    """
    Stronger check: run the same problem in serial (numpy reductions) and in
    whatever comm the fixture provides, and verify that both return the same
    minimiser when the comm-version's local slices are concatenated.
    """
    pnp = Reduction(comm)
    N_global = 256
    rng = np.random.default_rng(42)
    a_global = np.abs(rng.normal(size=N_global)) + 0.1
    y_global = rng.uniform(size=N_global)
    target = 0.4 * a_global.sum()

    # Serial reference, run on every rank independently against the global
    # arrays with plain numpy reductions.
    lc_serial = LinearConstraint(a_global, target, pnp=np)

    def fun_grad_serial(x):
        diff = x - y_global
        return 0.5 * float(np.sum(diff * diff)), diff

    res_serial = l_bfgs_projected(
        fun_grad_serial,
        np.full(N_global, 0.5),
        lc_serial,
        jac=True,
        bounds_lo=0.0,
        bounds_hi=1.0,
        gtol=1e-10,
    )
    assert res_serial.success, res_serial.message

    # Distributed run.
    a_local = _distribute(a_global, comm)
    y_local = _distribute(y_global, comm)
    lc_par = LinearConstraint(a_local, target, pnp=pnp)

    def fun_grad_par(x):
        diff = x - y_local
        return 0.5 * pnp.sum(diff * diff).item(), diff

    res_par = l_bfgs_projected(
        fun_grad_par,
        np.full_like(y_local, 0.5),
        lc_par,
        jac=True,
        bounds_lo=0.0,
        bounds_hi=1.0,
        comm=comm,
        gtol=1e-10,
    )
    parallel_assert(comm, res_par.success, res_par.message)

    # Compare distributed result to the serial reference's local slice.
    x_serial_local = _distribute(res_serial.x, comm)
    assert_all_allclose(comm, res_par.x, x_serial_local, atol=1e-7)
