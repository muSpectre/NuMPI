import numpy as np
import pytest

from NuMPI.Testing.Assertions import assert_all_allclose, parallel_assert

try:
    import scipy.optimize

    _scipy_present = True
except ModuleNotFoundError:
    _scipy_present = False

from test.Optimization.MPIMinimizationProblems import MPI_Quadratic

from NuMPI.Optimization import LinearConstraint
from NuMPI.Optimization.CCGWithoutRestart import \
    constrained_conjugate_gradients
from NuMPI.Tools import Reduction


def test_bugnicourt_cg(comm):
    n = 128

    obj = MPI_Quadratic(n, pnp=Reduction(comm), )

    xstart = np.random.normal(size=obj.nb_subdomain_grid_pts)

    res = constrained_conjugate_gradients(
        obj.f_grad,
        obj.hessian_product,
        args=(2,),
        x0=xstart,
        communicator=comm
    )
    parallel_assert(comm, res.success, res.message)
    print(res.nit)


@pytest.mark.skipif(not _scipy_present, reason='scipy not present')
def test_bugnicourt_cg_arbitrary_bounds(comm):
    n = 128
    np.random.seed(0)
    obj = MPI_Quadratic(n, pnp=Reduction(comm), )

    xstart = np.random.normal(size=obj.nb_subdomain_grid_pts)
    bounds = np.random.normal(size=obj.nb_subdomain_grid_pts)

    res = constrained_conjugate_gradients(
        obj.f_grad,
        obj.hessian_product,
        args=(2,),
        x0=xstart,
        communicator=comm,
        bounds=bounds,
        gtol=1e-8
    )
    parallel_assert(comm, res.success, res.message)

    parallel_assert(comm, (res.x >= bounds).all())
    print(res.nit)

    # TODO: we are not checking yet that the result is the same in parallel.
    if comm.size == 1:
        bnds = tuple([(b, None) for b in bounds])

        res_ref = scipy.optimize.minimize(
            obj.f_grad,
            args=(2,),
            x0=xstart,
            bounds=bnds,
            method="L-BFGS-B",
            jac=True,
            options=dict(gtol=1e-8, ftol=0))
        parallel_assert(comm, res_ref.success, res_ref.message)

        assert_all_allclose(comm, res.x, res_ref.x, atol=1e-6)


def test_bugnicourt_cg_active_bounds(comm):
    n = 128
    np.random.seed(0)
    obj = MPI_Quadratic(n, pnp=Reduction(comm), xmin=np.random.normal(size=n))

    xstart = np.random.normal(size=obj.nb_subdomain_grid_pts)

    res = constrained_conjugate_gradients(
        obj.f_grad,
        obj.hessian_product,
        args=(2,),
        x0=xstart,
        communicator=comm
    )
    parallel_assert(comm, res.success, res.message)
    print(res.nit)
    print(np.count_nonzero(res.x == 0))


def test_bugnicourt_cg_mean_val(comm):
    n = 128

    obj = MPI_Quadratic(n, pnp=Reduction(comm), )

    xstart = np.random.normal(size=obj.nb_subdomain_grid_pts)

    res = constrained_conjugate_gradients(
        obj.f_grad,
        obj.hessian_product,
        args=(2,),
        x0=xstart,
        mean_val=1.,
        communicator=comm
    )
    parallel_assert(comm, res.success, res.message)
    print(res.nit)


def test_bugnicourt_cg_mean_val_active_bounds(comm):
    n = 128
    np.random.seed(0)
    obj = MPI_Quadratic(n, pnp=Reduction(comm), xmin=np.random.normal(size=n))

    xstart = np.random.normal(size=obj.nb_subdomain_grid_pts)

    res = constrained_conjugate_gradients(
        obj.f_grad,
        obj.hessian_product,
        args=(2,),
        x0=xstart,
        mean_val=1.,
        communicator=comm
    )
    parallel_assert(comm, res.success, res.message)
    print(res.nit)


# ----------------------------------------------------------------------------
# Generic LinearConstraint path — the weighted generalization of mean_val.
# ----------------------------------------------------------------------------

def test_bugnicourt_cg_linear_constraint_uniform_matches_mean_val(comm):
    """
    A LinearConstraint with uniform weights and target = mean_val * N must
    produce the same minimiser as the legacy mean_val kwarg.
    """
    n = 128
    np.random.seed(0)
    obj = MPI_Quadratic(n, pnp=Reduction(comm))

    xstart = np.random.normal(size=obj.nb_subdomain_grid_pts)
    ones = np.ones_like(xstart)
    nb_DOF = Reduction(comm).sum(xstart.size) if comm is not None else xstart.size

    res_legacy = constrained_conjugate_gradients(
        obj.f_grad, obj.hessian_product, args=(2,),
        x0=xstart, mean_val=0.5, communicator=comm,
    )
    parallel_assert(comm, res_legacy.success, res_legacy.message)

    lc = LinearConstraint(
        ones, target=0.5 * nb_DOF,
        pnp=Reduction(comm) if comm is not None else np,
    )
    res_generic = constrained_conjugate_gradients(
        obj.f_grad, obj.hessian_product, args=(2,),
        x0=xstart, linear_constraint=lc, communicator=comm,
    )
    parallel_assert(comm, res_generic.success, res_generic.message)

    # Both paths should agree on the minimiser.
    assert_all_allclose(comm, res_legacy.x, res_generic.x, atol=1e-6)


def test_bugnicourt_cg_linear_constraint_non_uniform(comm):
    """
    Non-uniform weights exercise the generalised path that was previously
    unreachable (the hard-coded mean-value logic only handled a = 1).
    """
    n = 128
    np.random.seed(2)
    obj = MPI_Quadratic(n, pnp=Reduction(comm))

    xstart = np.random.normal(size=obj.nb_subdomain_grid_pts)
    a_local = np.abs(np.random.normal(size=xstart.size)) + 0.1
    # target chosen to be achievable (roughly half the "maximum" total load).
    if comm is None:
        total_a = a_local.sum()
    else:
        total_a = Reduction(comm).sum(a_local.sum())
    target = 0.5 * total_a

    lc = LinearConstraint(
        a_local, target=target,
        pnp=Reduction(comm) if comm is not None else np,
    )
    res = constrained_conjugate_gradients(
        obj.f_grad, obj.hessian_product, args=(2,),
        x0=xstart, linear_constraint=lc, communicator=comm,
    )
    parallel_assert(comm, res.success, res.message)

    # Constraint must hold to high accuracy at the solution.
    if comm is None:
        dot = np.dot(a_local, res.x.flatten())
    else:
        dot = Reduction(comm).sum(a_local * res.x.flatten())
    assert abs(dot - target) < 1e-6


def test_bugnicourt_cg_rejects_both_mean_val_and_linear_constraint(comm):
    n = 128
    obj = MPI_Quadratic(n, pnp=Reduction(comm))
    xstart = np.zeros(obj.nb_subdomain_grid_pts)
    lc = LinearConstraint(
        np.ones_like(xstart), target=1.0,
        pnp=Reduction(comm) if comm is not None else np,
    )
    import pytest
    with pytest.raises(ValueError):
        constrained_conjugate_gradients(
            obj.f_grad, obj.hessian_product, args=(2,),
            x0=xstart, mean_val=1.0, linear_constraint=lc,
            communicator=comm,
        )


def test_bugnicourt_cg_accepts_gradient_only_fun_with_jac_false(comm):
    np.random.seed(3)
    n = 64
    obj = MPI_Quadratic(n, pnp=Reduction(comm))

    xstart = np.random.normal(size=obj.nb_subdomain_grid_pts)

    res = constrained_conjugate_gradients(
        obj.grad,
        obj.hessian_product,
        args=(2,),
        jac=False,
        x0=xstart,
        communicator=comm,
    )
    parallel_assert(comm, res.success, res.message)


def test_bugnicourt_cg_accepts_split_fun_and_jac_callable(comm):
    np.random.seed(4)
    n = 64
    obj = MPI_Quadratic(n, pnp=Reduction(comm))

    xstart = np.random.normal(size=obj.nb_subdomain_grid_pts)

    res = constrained_conjugate_gradients(
        obj.f,
        obj.hessian_product,
        args=(2,),
        jac=obj.grad,
        x0=xstart,
        communicator=comm,
    )
    parallel_assert(comm, res.success, res.message)


def test_bugnicourt_cg_forwards_args_to_hessp(comm):
    np.random.seed(1)
    n = 32
    xstart = np.random.normal(size=n)
    target = np.random.normal(size=n)
    curvature = 3.0

    if comm is not None:
        rank = comm.Get_rank()
        size = comm.Get_size()
        step = n // size
        if rank == size - 1:
            sub = slice(rank * step, None)
        else:
            sub = slice(rank * step, (rank + 1) * step)
        xstart = xstart[sub]
        target = target[sub]

    def f_grad(x, scale):
        diff = x - target
        return 0.5 * scale * np.dot(diff, diff), scale * diff

    def hessp(d, scale):
        return scale * d

    res = constrained_conjugate_gradients(
        f_grad,
        hessp,
        x0=xstart,
        args=(curvature,),
        communicator=comm,
        bounds=np.full_like(xstart, -np.inf),
        gtol=1e-10,
    )

    parallel_assert(comm, res.success, res.message)
    assert_all_allclose(comm, res.x, target, atol=1e-7)
