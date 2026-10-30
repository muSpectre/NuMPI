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
Bound-constrained trust-region Newton-CG with optional index pinning and
optional inexact-evaluation (noise) control.

Feasible set (same as :func:`~NuMPI.Optimization.BoundedLBFGS.l_bfgs_bounded`):

    F = { x : lo <= x <= hi, x[zero_mask] = 0 }

Each iteration:

1. Build the box-active-masked gradient ``r_free`` (components where ``x``
   sits on an active bound *and* the gradient points further into the bound
   are zeroed; pinned indices are always zeroed). Its infinity norm is the
   convergence measure -- identical to ``l_bfgs_bounded``, so runs are
   directly comparable.
2. Approximately minimize the quadratic model
   ``m(s) = r_freeᵀ s + ½ sᵀ B s`` over the trust region ``||s|| <= Delta``,
   restricted to the free subspace, with the Steihaug-Toint truncated CG
   (Nocedal & Wright, Algorithm 7.2). The model Hessian ``B`` is supplied
   matrix-free via ``hessp`` and only needs to be *bounded* -- inexact
   Hessian-vector products (e.g. from truncated inner solves of an outer
   PDE-constrained problem) do not break global convergence; negative
   curvature is exploited by stepping to the trust-region boundary.
3. Accept or reject the projected trial point ``x_try = Pi_F(x + s)`` by the
   reduction ratio ``rho = (f - f_try) / pred`` and update the radius
   (reject below ``eta``; shrink x1/4 below rho=0.25; grow x2, capped, above
   rho=0.75 when the step reached the boundary). A boundary-limited step is
   never counted as converged.

**Robustness to inexact (noisy) objectives.** Unlike a backtracking line
search -- whose sufficient-decrease test must resolve decreases that shrink
``∝ alpha`` towards zero and hence *always* drowns in evaluation noise
eventually -- the trust-region test compares the actual reduction against the
*computable* predicted reduction ``pred``. When the caller can bound its own
evaluation error (e.g. via an adjoint-weighted residual of a truncated inner
solve), pass ``fun_error``/``request_accuracy``: the driver then enforces the
standard inexact-trust-region condition ``|f_err| <= eta_f * pred``
(Conn, Gould & Toint 2000, ch. 8.4/10.6; Kouri et al., SISC 2013/14),
requesting tighter inner accuracy exactly when the noise would corrupt the
acceptance test. With both hooks ``None`` this is a classic exact
trust-region method.

MPI-parallel: ``x``/gradient/bounds/``hessp`` operate on *local* per-rank
slices; the scalar objective must be globally reduced; all reductions go
through the caller-supplied ``pnp``.
"""

import logging
import math

import numpy as np
from array_api_compat import array_namespace
from array_api_compat import device as array_device

from ..Tools import Reduction
from ._lbfgs_helpers import _kkt_residual
from .Result import OptimizeResult

_log = logging.getLogger(__name__)


def _free_mask(grad, x, lo, hi, zero_mask, tol_box=1e-12):
    """Boolean mask of the *free* variables: not pinned, and not sitting on an
    active box face with the gradient pointing further into it (the complement
    of the components :func:`_kkt_residual` zeroes)."""
    xp = array_namespace(x)
    free = xp.ones(x.shape, dtype=xp.bool, device=array_device(x))
    if zero_mask is not None:
        free &= ~zero_mask
    if lo is not None:
        free &= ~((x <= lo + tol_box) & (grad >= 0.0))
    if hi is not None:
        free &= ~((x >= hi - tol_box) & (grad <= 0.0))
    return free


def _to_boundary(s, d, sds, dds, sdd, delta):
    """Both roots ``tau`` of ``||s + tau d||^2 = delta^2``.

    ``sds = sᵀs``, ``dds = dᵀd``, ``sdd = sᵀd`` are the (already reduced)
    dot products. Returns ``(tau_minus, tau_plus)`` with
    ``tau_minus <= tau_plus``. Python floats, not NumPy scalars: a NumPy
    scalar times an array of another namespace would try to convert the
    array to NumPy."""
    tmp = math.sqrt(max(sdd * sdd - dds * (sds - delta * delta), 0.0))
    return (-sdd - tmp) / dds, (-sdd + tmp) / dds


class _HostScalars:
    """The global reductions of `pnp` as Python floats: in the iteration only
    these single numbers cross to the host, whatever the arrays' namespace
    (numpy, cupy, ...) and device."""

    def __init__(self, pnp):
        self._pnp = pnp

    def sum(self, a):
        return float(self._pnp.sum(a))

    def max(self, a):
        return float(self._pnp.max(a))


def _steihaug_cg(hessp_free, g_free, delta, tol, maxiter, pnp):
    """Steihaug-Toint truncated CG on ``min m(s) = gᵀs + ½ sᵀBs`` s.t.
    ``||s|| <= delta`` (Nocedal & Wright, Algorithm 7.2).

    ``hessp_free`` must map a free-subspace vector to ``B v`` masked to the
    free subspace. Returns ``(s, pred, on_boundary, status, nb_hessp)`` with
    ``pred = m(0) - m(s) = -m(s) >= 0`` the predicted reduction.

    Three exits: residual tolerance (interior step), negative curvature or
    trust-region crossing (both: step to the boundary). For negative
    curvature both sphere intersections are evaluated and the one with the
    smaller model value is taken (the muSpectre convention); for a
    trust-region crossing during a positive-curvature step the positive root
    is the minimizer along ``d``.
    """
    xp = array_namespace(g_free)
    s = xp.zeros_like(g_free)
    Bs = xp.zeros_like(g_free)  # running B s, for cheap model evaluations
    r = xp.asarray(g_free, copy=True)  # residual of grad m = g + B s at s = 0
    d = -r
    rr = pnp.sum(r * r)
    g_norm = math.sqrt(rr)
    if g_norm == 0.0:
        return s, 0.0, False, "zero gradient", 0
    rtol2 = (tol * g_norm) ** 2

    def model(sv, Bsv):
        # m(s) = gᵀ s + ½ sᵀ B s, all-reduced.
        return pnp.sum(g_free * sv) + 0.5 * pnp.sum(sv * Bsv)

    nb_hessp = 0
    for _ in range(maxiter):
        Bd = hessp_free(d)
        nb_hessp += 1
        dBd = pnp.sum(d * Bd)
        sds = pnp.sum(s * s)
        sdd = pnp.sum(s * d)
        dds = pnp.sum(d * d)
        if dBd <= 0.0:
            # Negative curvature: the model is unbounded along d; go to the
            # sphere. Evaluate both intersections, keep the smaller model.
            tau_m, tau_p = _to_boundary(s, d, sds, dds, sdd, delta)
            cand = []
            for tau in (tau_m, tau_p):
                sv = s + tau * d
                Bsv = Bs + tau * Bd
                cand.append((model(sv, Bsv), sv, Bsv))
            m_val, s, Bs = min(cand, key=lambda c: c[0])
            return s, -m_val, True, "negative curvature", nb_hessp
        alpha = rr / dBd
        # Would the CG step leave the trust region?
        step_norm2 = sds + 2.0 * alpha * sdd + alpha * alpha * dds
        if step_norm2 >= delta * delta:
            _, tau = _to_boundary(s, d, sds, dds, sdd, delta)
            s = s + tau * d
            Bs = Bs + tau * Bd
            return s, -model(s, Bs), True, "boundary", nb_hessp
        s = s + alpha * d
        Bs = Bs + alpha * Bd
        r = r + alpha * Bd
        new_rr = pnp.sum(r * r)
        if new_rr <= rtol2:
            return s, -model(s, Bs), False, "tolerance", nb_hessp
        d = -r + (new_rr / rr) * d
        rr = new_rr
    return s, -model(s, Bs), False, "maxiter", nb_hessp


def tr_newton_bounded(
    fun,
    x0,
    hessp,
    args=(),
    jac=None,
    bounds_lo=None,
    bounds_hi=None,
    zero_mask=None,
    gtol=1e-5,
    maxiter=200,
    delta0=0.05,
    delta_max=0.5,
    eta=0.1,
    inner_tol=None,
    inner_maxiter=None,
    hessp_tol=False,
    fun_error=None,
    request_accuracy=None,
    eta_f=0.25,
    max_accuracy_retries=5,
    comm=None,
    pnp=None,
    callback=None,
    disp=False,
):
    """
    Bound-constrained trust-region Newton-CG with optional index pinning and
    optional inexact-evaluation control.

    Parameters
    ----------
    fun : callable
        Objective. If ``jac`` is ``True`` (or ``None``), ``fun(x, *args)``
        must return ``(energy, gradient)``. If ``jac`` is a callable,
        ``fun`` returns only the energy and ``jac(x, *args)`` returns the
        gradient.
    x0 : array
        Initial guess (any shape; local per-rank slice) of any array-API
        namespace, e.g. a NumPy or a CuPy array. The iteration runs in that
        namespace and on that device: ``fun``, ``hessp`` and ``callback``
        receive such arrays, and the result's ``x`` and ``jac`` are such
        arrays; only reduced scalars come to the host. Projected onto the
        feasible set before the loop starts.
    hessp : callable
        Matrix-free model Hessian: ``hessp(x, v, *args)`` returns ``B v`` for
        a direction ``v`` shaped like ``x`` (local slices in and out). ``B``
        only needs to be symmetric and bounded -- an approximate or inexactly
        evaluated Hessian slows local convergence but does not break global
        convergence.
    args : tuple, optional
        Extra positional arguments forwarded to ``fun`` / ``jac`` / ``hessp``.
    jac : bool or callable, optional
        See ``fun``. Default ``None`` means ``jac=True`` (fused).
    bounds_lo, bounds_hi : array_like or scalar, optional
        Box bounds. ``None`` means unbounded on that side.
    zero_mask : array_like of bool, optional
        Indices pinned to ``x = 0`` (e.g. contact nodes).
    gtol : float, optional
        Convergence tolerance on the infinity norm of the box-masked
        gradient (same measure as ``l_bfgs_bounded``). Default 1e-5.
    maxiter : int, optional
        Maximum number of outer iterations. Default 200.
    delta0, delta_max : float, optional
        Initial and maximum trust-region radius in **RMS-per-component
        units**: the Euclidean radius is ``delta * sqrt(N_global)``, so the
        same value expresses "typical change per variable" at any problem
        size. Defaults 0.05 and 0.5.
    eta : float, optional
        Step acceptance threshold: accept iff ``rho >= eta``. Default 0.1.
    inner_tol : float, optional
        Relative residual tolerance of the Steihaug CG. ``None`` (default)
        uses the superlinear forcing sequence
        ``min(0.5, sqrt(||g_free|| / ||g_free,0||))`` (Nocedal & Wright,
        p. 169), with the norms relative to that at the first iterate so
        that the sequence does not depend on the units of the objective.
    inner_maxiter : int, optional
        Iteration cap of the Steihaug CG. Default: number of global degrees
        of freedom (i.e. effectively uncapped).
    hessp_tol : bool, optional
        If ``True``, ``hessp`` is called as ``hessp(x, v, *args, tol=tol)``
        with ``tol`` the relative residual tolerance of the Steihaug CG that
        requests the product. A Hessian-vector product that is itself
        evaluated inexactly (e.g. by an inner iterative solve) can then be
        made only as accurate as the subproblem needs. Default ``False``.
    fun_error : callable, optional
        ``fun_error()`` returns a computable bound on the absolute error of
        the most recent ``fun`` evaluation (e.g. an adjoint-weighted residual
        of a truncated inner solve). Enables the inexact-trust-region
        accuracy control.
    request_accuracy : callable, optional
        ``request_accuracy(target)`` asks the caller to tighten its inner
        solves until the ``fun`` evaluation error is below the absolute
        value ``target``. Called when ``fun_error() > eta_f * pred``; the
        current and trial points are then re-evaluated.
    eta_f : float, optional
        Fraction of the predicted reduction the evaluation error must stay
        below (``|f_err| <= eta_f * pred``). Default 0.25.
    max_accuracy_retries : int, optional
        Maximum tighten-and-re-evaluate rounds per iteration. Default 5.
    comm, pnp : MPI.Comm or Reduction, optional
        MPI communicator or pre-built reduction wrapper. Pass one, not both;
        without either the run is serial. ``pnp.sum``/``pnp.max`` must
        accept arrays of the namespace of ``x0`` (:class:`Reduction` does).
    callback : callable, optional
        ``callback(x)`` called after each *accepted* iterate.
    disp : bool, optional
        Print a per-iteration diagnostic table.

    Returns
    -------
    OptimizeResult
        With ``x, fun, jac, nit, success, message, max_grad`` plus the
        trust-region diagnostics ``nb_hessp`` (total Hessian products),
        ``delta_history``, ``rho_history`` (per outer iteration; ``rho`` is
        ``nan`` where the model predicted no reduction).

    Notes
    -----
    **MPI contract.** As for ``l_bfgs_bounded``: ``x``, the gradient, the
    bounds, ``zero_mask`` and the vectors seen by ``hessp`` are all *local*
    per-rank slices of the global problem; the scalar energy returned by
    ``fun`` (and the error bound returned by ``fun_error``) must be
    **globally reduced** (identical on every rank).
    """
    if comm is not None:
        if pnp is not None:
            raise RuntimeError("Please specify either `comm` or `pnp`, not both.")
        pnp = Reduction(comm)
    elif pnp is None:
        pnp = Reduction()
    red = _HostScalars(pnp)

    # The iteration runs in the namespace and on the device of x0 (numpy,
    # cupy, ...): the iterate, the gradient, the free mask, the bounds and
    # the Steihaug vectors stay there, and fun and hessp get and return such
    # arrays. Only the reduced scalars come to the host.
    if not hasattr(x0, "shape"):
        x0 = np.asarray(x0)
    xp = array_namespace(x0)
    dev = array_device(x0)
    original_shape = tuple(x0.shape)

    def _flat(a):
        if a is None or np.isscalar(a):
            return a
        return xp.reshape(xp.asarray(a, device=dev), (-1,))

    bounds_lo = _flat(bounds_lo)
    bounds_hi = _flat(bounds_hi)
    zero_mask = (
        None if zero_mask is None else xp.astype(_flat(zero_mask), xp.bool)
    )

    def _shaped(a):
        return xp.reshape(a, original_shape)

    def _flat_result(a):
        return xp.reshape(xp.asarray(a), (-1,))

    if jac is True or jac is None:

        def fun_grad(x):
            phi, g = fun(_shaped(x), *args)
            return float(phi), _flat_result(g)

    elif jac is False:
        raise NotImplementedError("Numerical evaluation of gradient not implemented")
    else:

        def fun_grad(x):
            xr = _shaped(x)
            return float(fun(xr, *args)), _flat_result(jac(xr, *args))

    def hessp_flat(x, v, tol):
        kwargs = {"tol": tol} if hessp_tol else {}
        return _flat_result(hessp(_shaped(x), _shaped(v), *args, **kwargs))

    def project(y):
        z = y
        if bounds_lo is not None or bounds_hi is not None:
            z = xp.clip(z, bounds_lo, bounds_hi)
        if zero_mask is not None:
            z = xp.where(zero_mask, 0.0, z)
        return z

    x = project(xp.reshape(xp.asarray(x0, dtype=xp.float64, device=dev),
                           (-1,)))

    # Global degree-of-freedom count: converts the RMS radius parameters to
    # Euclidean radii and caps the inner CG.
    n_global = int(red.sum(np.full(1, float(math.prod(x.shape)))))
    delta = float(delta0) * math.sqrt(n_global)
    delta_cap = float(delta_max) * math.sqrt(n_global)
    if inner_maxiter is None:
        inner_maxiter = n_global

    phi, grad = fun_grad(x)

    nb_hessp_total = 0
    # Projected-gradient norm at the first iterate, the reference of the
    # forcing sequence
    r_norm0 = None
    delta_history = []
    rho_history = []

    def _finish(success, nit, r_norm, message):
        return OptimizeResult(
            {
                "success": bool(success),
                "x": _shaped(x),
                "fun": phi,
                "jac": _shaped(grad),
                "nit": int(nit),
                "message": message,
                "max_grad": float(r_norm),
                "nb_hessp": int(nb_hessp_total),
                "delta_history": delta_history,
                "rho_history": rho_history,
            }
        )

    if disp and (comm is None or pnp is np or comm.rank == 0):
        print(
            f"{'iter':<5} {'f':<14} {'|r|inf_free':<14} {'Delta_rms':<10} "
            f"{'rho':<10} {'m_cg':<5} {'step':<6}"
        )
        print("-" * 68)

    for iteration in range(1, maxiter + 1):
        r_free = _kkt_residual(
            grad, x, lo=bounds_lo, hi=bounds_hi, zero_mask=zero_mask
        )
        r_norm = red.max(xp.abs(r_free))
        # First-order optimality, tested unconditionally (Conn, Gould &
        # Toint, Alg. 6.1.1). The KKT residual is evaluated at the current
        # iterate and does not depend on the trust region, so a step that
        # ended on the trust-region boundary is no evidence against
        # stationarity: where the curvature along the gradient is near zero
        # every step runs to the boundary however small the gradient, and
        # gating this test on an interior step would leave such a run unable
        # to stop at all. See test_converges_when_steps_stay_boundary_limited.
        if r_norm < gtol:
            return _finish(
                True, iteration - 1, r_norm,
                "CONVERGENCE: NORM_OF_PROJECTED_GRADIENT_<=_GTOL",
            )

        free = _free_mask(grad, x, bounds_lo, bounds_hi, zero_mask)

        # The forcing sequence on the *relative* projected gradient: on
        # ||g|| itself it would depend on the units of the objective -- for
        # an objective scaled by 1/N, it would demand a nearly exact
        # subproblem solve from the first iteration on.
        if r_norm0 is None:
            r_norm0 = r_norm
        tol_k = (
            min(0.5, math.sqrt(r_norm / r_norm0)) if r_norm0 > 0 else 0.0
        ) if inner_tol is None else inner_tol

        def hessp_free(v):
            return xp.where(
                free, hessp_flat(x, xp.where(free, v, 0.0), tol_k), 0.0)
        s, pred, on_boundary, status, nb_hessp = _steihaug_cg(
            hessp_free, r_free, delta, tol_k, inner_maxiter, red
        )
        nb_hessp_total += nb_hessp

        if pred <= 0.0:
            # The model predicts no reduction (numerically degenerate
            # subproblem). Shrink and retry; if the region collapses, stop.
            delta *= 0.25
            delta_history.append(delta / math.sqrt(n_global))
            rho_history.append(np.nan)
            if delta < 1e-14 * delta_cap:
                return _finish(
                    False, iteration, r_norm,
                    "NO CONVERGENCE: trust region collapsed",
                )
            continue

        x_try = project(x + s)
        # Projection may clip the step at the box; the model reduction must
        # be measured for the step actually taken.
        s_eff = x_try - x
        if red.max(xp.abs(s_eff - s)) > 0.0:
            Bs = hessp_free(s_eff)
            nb_hessp_total += 1
            pred = -(
                red.sum(r_free * s_eff) + 0.5 * red.sum(s_eff * Bs)
            )
            if pred <= 0.0:
                delta *= 0.25
                delta_history.append(delta / math.sqrt(n_global))
                rho_history.append(np.nan)
                if delta < 1e-14 * delta_cap:
                    return _finish(
                        False, iteration, r_norm,
                        "NO CONVERGENCE: trust region collapsed",
                    )
                continue

        phi_try, grad_try = fun_grad(x_try)

        # Inexact-evaluation control: the evaluation error must stay below a
        # fraction of the predicted reduction, or the acceptance ratio is
        # noise (Conn-Gould-Toint 8.4; Kouri et al. 2013/14). The caller
        # tightens its inner solves and both points are re-evaluated. If a
        # retry fails to reduce the reported error, the caller has hit its
        # accuracy floor (e.g. the attainable residual of a single-precision
        # solve) -- stop retrying and let the rho-test decide with the best
        # available accuracy.
        if fun_error is not None and request_accuracy is not None:
            prev_err = None
            for _ in range(max_accuracy_retries):
                err = fun_error()
                if err <= eta_f * pred:
                    break
                if prev_err is not None and err > 0.9 * prev_err:
                    break  # no further improvement possible
                prev_err = err
                request_accuracy(eta_f * pred)
                phi, grad = fun_grad(x)
                phi_try, grad_try = fun_grad(x_try)

        ared = phi - phi_try
        # Round-off safeguard (Conn-Gould-Toint 17.4.2): near convergence the
        # model reduction drops below the round-off floor of f itself; the
        # ratio is then unresolvable noise and would reject good steps until
        # the region collapses. If pred is below that floor and the actual
        # value did not measurably increase, trust the model (rho = 1) -- the
        # gradient test still guards convergence.
        eps_f = 10.0 * np.finfo(float).eps * (1.0 + abs(phi))
        if pred <= eps_f and ared >= -eps_f:
            rho = 1.0
        else:
            rho = ared / pred

        # Radius update (muSpectre blueprint): shrink hard on a bad model,
        # grow only when the model is good AND the boundary was limiting.
        if rho < 0.25:
            delta *= 0.25
        elif rho > 0.75 and on_boundary:
            delta = min(2.0 * delta, delta_cap)
        delta_history.append(delta / math.sqrt(n_global))
        rho_history.append(float(rho))

        accepted = rho >= eta
        if disp and (comm is None or comm.rank == 0):
            print(
                f"{iteration:<5d} {phi:<14.6e} {r_norm:<14.4e} "
                f"{delta / math.sqrt(n_global):<10.3e} {rho:<10.3g} "
                f"{nb_hessp:<5d} {'acc' if accepted else 'rej':<6}"
            )

        if accepted:
            x, phi, grad = x_try, phi_try, grad_try
            if callback is not None:
                callback(_shaped(x))
        else:
            if delta < 1e-14 * delta_cap:
                return _finish(
                    False, iteration, r_norm,
                    "NO CONVERGENCE: trust region collapsed",
                )

    r_free = _kkt_residual(
        grad, x, lo=bounds_lo, hi=bounds_hi, zero_mask=zero_mask
    )
    r_norm = red.max(xp.abs(r_free))
    if r_norm < gtol:
        return _finish(
            True, maxiter, r_norm,
            "CONVERGENCE: NORM_OF_PROJECTED_GRADIENT_<=_GTOL",
        )
    return _finish(
        False, maxiter, r_norm, "NO CONVERGENCE: MAXITERATIONS REACHED"
    )
