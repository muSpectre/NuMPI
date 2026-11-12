#
# Copyright 2026 Yizhen Wang
#
# ### MIT license
#

import numpy as np
import pytest

from NuMPI.Optimization import QuadraticConstraint, l_bfgs_qc_projected
from NuMPI.Testing.Assertions import assert_all_allclose, parallel_assert
from NuMPI.Tools import Reduction

# min 0.5||x - y||^2 on the sphere 0.5||x||^2 = 0.5r^2, with closed-form minimisers.


def _local(global_arr, comm):
    return np.array_split(global_arr, comm.Get_size())[comm.Get_rank()].copy()


def _sphere_with_bounds(comm, pnp):
    # x >= 0: x* = r y+ / ||y+||, lambda* = 1 - ||y+|| / r.
    n = 40
    r = 2.0
    y = np.random.default_rng(0).normal(size=n)
    y_pos = np.maximum(y, 0.0)
    x_true = r * y_pos / np.linalg.norm(y_pos)
    lam_true = np.array([1.0 - np.linalg.norm(y_pos) / r])
    k = _local(y, comm).size

    def hessp(u):
        return u

    qc = QuadraticConstraint(np.zeros(k), hessp, 0.5 * r**2, pnp=pnp)
    x0 = np.full(k, r / np.sqrt(n))
    return y, qc, x0, 0.0, x_true, lam_true


def _sphere_and_plane(comm, pnp):
    # sum(x) = s on the sphere: x* is the point of the circle nearest to y.
    n = 40
    r = 2.0
    s = 1.0
    y = np.random.default_rng(1).normal(size=n)
    centre = np.full(n, s / n)
    rho = np.sqrt(r**2 - s**2 / n)
    p = y - (y.mean() - s / n)
    x_true = centre + rho * (p - centre) / np.linalg.norm(p - centre)
    lam_true = np.linalg.lstsq(
        np.column_stack([np.ones(n), x_true]), x_true - y, rcond=None
    )[0]
    k = _local(y, comm).size

    def hessp(u):
        return np.stack([np.zeros_like(u), u])

    qc = QuadraticConstraint(
        np.stack([np.ones(k), np.zeros(k)]), hessp, [s, 0.5 * r**2], pnp=pnp
    )
    return y, qc, _local(y, comm), None, x_true, lam_true


@pytest.mark.parametrize(
    "case", [_sphere_with_bounds, _sphere_and_plane], ids=["sphere_bounds", "sphere_plane"]
)
def test_returns_closed_form_minimiser_and_multipliers(comm, case):
    pnp = Reduction(comm)
    y, qc, x0, lo, x_true, lam_true = case(comm, pnp)
    y_local = _local(y, comm)

    def fun_grad(x):
        diff = x - y_local
        return 0.5 * pnp.sum(diff * diff).item(), diff

    res = l_bfgs_qc_projected(
        fun_grad, x0, qc, bounds_lo=lo, comm=comm, gtol=1e-10
    )

    parallel_assert(comm, res.success, res.message)
    assert_all_allclose(comm, res.x, _local(x_true, comm), atol=1e-8)
    parallel_assert(
        comm, np.allclose(res.multiplier, lam_true, atol=1e-8), str(res.multiplier)
    )
    parallel_assert(comm, res.constr_violation < 1e-12, str(res.constr_violation))
