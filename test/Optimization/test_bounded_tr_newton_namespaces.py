"""
tr_newton_bounded on arrays of other array-API namespaces.

Each test runs on three backends:

* numpy;
* array_api_strict on its second device, ``Device("device1")``: arrays that,
  like cupy's, refuse conversion to NumPy and mixing with arrays of another
  device, so that every implicit host copy or array created on the wrong
  device fails -- on the CPU, in every CI run;
* cupy, when a GPU is present (``pip install NuMPI[gpu]``).

The iteration must stay in the namespace and on the device of ``x0``: ``fun``
and ``hessp`` get such arrays, and the result is one; only reduced scalars
reach the host.
"""

import os

import numpy as np
import pytest

from NuMPI.Optimization import tr_newton_bounded
from NuMPI.Testing.Assertions import assert_all_allclose, parallel_assert
from NuMPI.Tools import Reduction


class Backend:
    """An array namespace with a device, and the conversions tests need."""

    def __init__(self, name):
        self.name = name
        if name == "numpy":
            self.xp = np
        elif name == "array_api_strict":
            import array_api_strict

            self.xp = array_api_strict
            self.device = array_api_strict.Device("device1")
            self.array_type = type(array_api_strict.asarray(0.0))
        elif name == "cupy":
            import cupy

            cupy.zeros(1)  # raises without a usable GPU
            self.xp = cupy
        else:
            raise ValueError(name)

    def asarray(self, a, dtype=None):
        """A NumPy array (or number) as an array of this backend."""
        a = np.asarray(a, dtype=dtype)
        if self.name == "array_api_strict":
            return self.xp.asarray(a, device=self.device)
        return self.xp.asarray(a)

    def to_numpy(self, a):
        if self.name == "array_api_strict":
            # Through DLPack: array_api_strict's to_device uses the copy
            # keyword of np.asarray, which NumPy 1.x does not have
            return np.from_dlpack(a)
        if self.name == "cupy":
            return a.get()
        return np.asarray(a)

    def check(self, a):
        """Assert that `a` is an array of this backend, on its device."""
        if self.name == "numpy":
            assert isinstance(a, np.ndarray)
        elif self.name == "array_api_strict":
            assert isinstance(a, self.array_type)
            assert a.device == self.device
        else:
            assert isinstance(a, self.xp.ndarray)


def _backend(name):
    try:
        return Backend(name)
    except Exception as err:  # not installed, or no GPU
        # The GPU CI job sets this, so that it cannot pass by skipping
        if name == "cupy" and os.environ.get("NUMPI_REQUIRE_CUPY"):
            raise
        pytest.skip(f"{name} unavailable: {err}")


@pytest.fixture(params=["numpy", "array_api_strict", "cupy"])
def be(request):
    return _backend(request.param)


def _quad(be, y, w=1.0):
    """``0.5 sum(w (x - y)^2)``; every argument is checked to be an array of
    the backend."""
    y = be.asarray(y)
    w = be.asarray(w) if np.ndim(w) else w

    def fun_grad(x):
        be.check(x)
        d = x - y
        return 0.5 * float(be.xp.sum(w * d * d)), w * d

    def hessp(x, v, tol=None):
        be.check(x)
        be.check(v)
        return w * v

    return fun_grad, hessp


def _indefinite(be, diag, g):
    diag, g = be.asarray(diag), be.asarray(g)

    def fun_grad(x):
        be.check(x)
        return (float(be.xp.sum(0.5 * diag * x * x - g * x)),
                diag * x - g)

    def hessp(x, v):
        be.check(v)
        return diag * v

    return fun_grad, hessp


def _indefinite_minimiser(diag, g):
    x = np.where(diag > 0, np.clip(g / np.where(diag > 0, diag, 1.0), -1, 1),
                 np.sign(g))
    return np.where((diag < 0) & (g == 0), 1.0, x)


def test_unbounded_quadratic(be):
    rng = np.random.default_rng(0)
    y = rng.normal(size=64)
    fun_grad, hessp = _quad(be, y)
    res = tr_newton_bounded(fun_grad, be.asarray(np.zeros(64)), hessp,
                            jac=True, gtol=1e-10, delta_max=10.0)
    assert res.success, res.message
    be.check(res.x)
    be.check(res.jac)
    np.testing.assert_allclose(be.to_numpy(res.x), y, atol=1e-8)


def test_box_matches_clip(be):
    rng = np.random.default_rng(1)
    N = 128
    y = rng.normal(size=N) * 0.6 + 0.3
    w = np.exp(rng.uniform(0, 6, size=N))
    fun_grad, hessp = _quad(be, y, w)
    res = tr_newton_bounded(fun_grad, be.asarray(np.full(N, 0.5)), hessp,
                            jac=True, bounds_lo=0.0, bounds_hi=1.0,
                            gtol=1e-10)
    assert res.success, res.message
    np.testing.assert_allclose(be.to_numpy(res.x), np.clip(y, 0.0, 1.0),
                               atol=1e-8)


def test_array_bounds(be):
    """Bounds given as arrays (here host NumPy arrays) move to the device of
    x0 once."""
    rng = np.random.default_rng(9)
    N = 64
    y = rng.normal(size=N)
    lo = np.full(N, -0.5)
    hi = rng.uniform(0.0, 1.0, size=N)
    fun_grad, hessp = _quad(be, y)
    res = tr_newton_bounded(fun_grad, be.asarray(np.zeros(N)), hessp,
                            jac=True, bounds_lo=lo, bounds_hi=hi, gtol=1e-10)
    assert res.success, res.message
    np.testing.assert_allclose(be.to_numpy(res.x), np.clip(y, lo, hi),
                               atol=1e-8)


def test_negative_curvature(be):
    rng = np.random.default_rng(2)
    N = 96
    diag = rng.uniform(-1.0, 2.0, size=N)
    diag[np.abs(diag) < 0.1] = 0.5
    g = rng.normal(size=N)
    fun_grad, hessp = _indefinite(be, diag, g)
    res = tr_newton_bounded(fun_grad, be.asarray(np.zeros(N)), hessp,
                            jac=True, bounds_lo=-1.0, bounds_hi=1.0,
                            gtol=1e-8, maxiter=500)
    assert res.success, res.message
    np.testing.assert_allclose(be.to_numpy(res.x),
                               _indefinite_minimiser(diag, g), atol=1e-6)


def test_zero_mask(be):
    rng = np.random.default_rng(4)
    N = 64
    y = rng.normal(size=N)
    mask = np.zeros(N, dtype=bool)
    mask[::3] = True
    fun_grad, hessp = _quad(be, y)
    res = tr_newton_bounded(fun_grad, be.asarray(np.zeros(N)), hessp,
                            jac=True, zero_mask=be.asarray(mask),
                            gtol=1e-10, delta_max=10.0)
    assert res.success, res.message
    x = be.to_numpy(res.x)
    np.testing.assert_allclose(x[mask], 0.0, atol=0.0)
    np.testing.assert_allclose(x[~mask], y[~mask], atol=1e-8)


def test_nd_x0(be):
    rng = np.random.default_rng(5)
    shape = (8, 8)
    y = rng.normal(size=shape)
    fun_grad, hessp = _quad(be, y)
    res = tr_newton_bounded(fun_grad, be.asarray(np.zeros(shape)), hessp,
                            jac=True, gtol=1e-10, delta_max=10.0)
    assert res.success, res.message
    assert tuple(res.x.shape) == shape and tuple(res.jac.shape) == shape
    np.testing.assert_allclose(be.to_numpy(res.x), y, atol=1e-8)


def test_callback_and_hessp_tol(be):
    rng = np.random.default_rng(7)
    N = 32
    y = rng.normal(size=N)
    w = np.logspace(0, 3, N)
    fun_grad, hessp = _quad(be, y, w)
    tols = []

    def hessp_recording(x, v, tol):
        tols.append(tol)
        return hessp(x, v)

    iterates = []

    def callback(x):
        be.check(x)
        iterates.append(be.to_numpy(x))

    res = tr_newton_bounded(fun_grad, be.asarray(np.zeros(N)),
                            hessp_recording, jac=True, gtol=1e-10,
                            delta_max=10.0, hessp_tol=True,
                            callback=callback)
    assert res.success, res.message
    assert iterates and len(tols) == res.nb_hessp
    assert all(isinstance(t, float) for t in tols)


def test_matches_numpy(be):
    """Same problem, same iterates as with NumPy."""
    rng = np.random.default_rng(10)
    N = 96
    diag = rng.uniform(-1.0, 2.0, size=N)
    diag[np.abs(diag) < 0.1] = 0.5
    g = rng.normal(size=N)
    runs = []
    for backend in (_backend("numpy"), be):
        fun_grad, hessp = _indefinite(backend, diag, g)
        res = tr_newton_bounded(fun_grad, backend.asarray(np.zeros(N)), hessp,
                                jac=True, bounds_lo=-1.0, bounds_hi=1.0,
                                gtol=1e-8, maxiter=500)
        runs.append((res.nit, res.nb_hessp, backend.to_numpy(res.x)))
    (nit_a, nh_a, x_a), (nit_b, nh_b, x_b) = runs
    assert (nit_a, nh_a) == (nit_b, nh_b)
    np.testing.assert_allclose(x_b, x_a, rtol=1e-12, atol=1e-14)


def test_accuracy_hooks(be):
    rng = np.random.default_rng(6)
    N = 64
    y = rng.normal(size=N) * 0.4
    w = np.exp(rng.uniform(0, 2, size=N))
    yb, wb = be.asarray(y), be.asarray(w)
    state = {"amp": 1e-1, "requests": 0}

    def fun_grad(x):
        be.check(x)
        d = x - yb
        f = 0.5 * float(be.xp.sum(wb * d * d))
        f_noise = state["amp"] * np.sin(1e3 * float(be.xp.sum(x)))
        g_noise = state["amp"] * be.xp.sin(1e3 * x + 1.0)
        return f + f_noise, wb * d + g_noise

    def request_accuracy(target):
        state["requests"] += 1
        state["amp"] = min(state["amp"], 0.5 * target)

    res = tr_newton_bounded(
        fun_grad, be.asarray(np.zeros(N)), lambda x, v: wb * v, jac=True,
        bounds_lo=-1.0, bounds_hi=1.0, gtol=1e-6, maxiter=500,
        fun_error=lambda: state["amp"], request_accuracy=request_accuracy)
    assert state["requests"] > 0
    assert res.success, res.message
    np.testing.assert_allclose(be.to_numpy(res.x), np.clip(y, -1.0, 1.0),
                               atol=1e-4)


def test_reduction_returns_host_scalars(be, comm):
    pnp = Reduction(comm)
    a = be.asarray(np.arange(1.0, 5.0) + comm.rank)
    n = comm.size
    total = sum(np.sum(np.arange(1.0, 5.0) + r) for r in range(n))
    for value, expected in [(pnp.sum(a), total),
                            (pnp.max(a), 4.0 + n - 1),
                            (pnp.min(a), 1.0),
                            (pnp.mean(a), total / (4 * n))]:
        # A host value, whatever the input's namespace
        assert isinstance(value, (np.ndarray, np.generic))
        assert np.ndim(value) == 0
        assert float(value) == pytest.approx(expected)
    assert bool(pnp.any(a > 3.5 + n - 1))
    assert not bool(pnp.all(a > 1.5))
    if be.name != "numpy":
        # Only full reductions leave the device, as single numbers
        with pytest.raises(NotImplementedError):
            pnp.sum(be.asarray(np.ones((2, 3))), axis=0)


def _distribute(global_arr, comm):
    return np.array_split(global_arr, comm.Get_size())[comm.Get_rank()]


def test_mpi_box(be, comm):
    pnp = Reduction(comm)
    N_global = 256
    rng = np.random.default_rng(0)
    y_global = rng.normal(size=N_global) * 0.6 + 0.3
    w_global = np.exp(rng.uniform(0, 4, size=N_global))
    y = _distribute(y_global, comm)
    w = _distribute(w_global, comm)
    yb, wb = be.asarray(y), be.asarray(w)

    def fun_grad(x):
        be.check(x)
        d = x - yb
        return float(pnp.sum(0.5 * wb * d * d)), wb * d

    res = tr_newton_bounded(fun_grad, be.asarray(np.full_like(y, 0.5)),
                            lambda x, v: wb * v, jac=True, bounds_lo=0.0,
                            bounds_hi=1.0, comm=comm, gtol=1e-10)
    parallel_assert(comm, res.success, res.message)
    assert_all_allclose(comm, be.to_numpy(res.x), np.clip(y, 0.0, 1.0),
                        atol=1e-8)


def test_mpi_negative_curvature(be, comm):
    pnp = Reduction(comm)
    N_global = 192
    rng = np.random.default_rng(1)
    diag_global = rng.uniform(-1.0, 2.0, size=N_global)
    diag_global[np.abs(diag_global) < 0.1] = 0.5
    g_global = rng.normal(size=N_global)
    diag = _distribute(diag_global, comm)
    g = _distribute(g_global, comm)
    db, gb = be.asarray(diag), be.asarray(g)

    def fun_grad(x):
        be.check(x)
        return (float(pnp.sum(0.5 * db * x * x - gb * x)), db * x - gb)

    res = tr_newton_bounded(fun_grad, be.asarray(np.zeros_like(g)),
                            lambda x, v: db * v, jac=True, bounds_lo=-1.0,
                            bounds_hi=1.0, comm=comm, gtol=1e-8, maxiter=500)
    parallel_assert(comm, res.success, res.message)
    assert_all_allclose(comm, be.to_numpy(res.x),
                        _indefinite_minimiser(diag, g), atol=1e-6)
