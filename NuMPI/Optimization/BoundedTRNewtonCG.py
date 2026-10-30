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
Bound-constrained trust-region Newton-CG with optional index pinning,
optional inexact-evaluation (noise) control, and optional preconditioning of
the inner Steihaug-Toint CG.

Feasible set (same as :func:`~NuMPI.Optimization.BoundedLBFGS.l_bfgs_bounded`):

    F = { x : lo <= x <= hi, x[zero_mask] = 0 }

Each iteration:

1. Build the box-active-masked gradient ``r_free`` (components where ``x``
   sits on an active bound *and* the gradient points further into the bound
   are zeroed; pinned indices are always zeroed). Its infinity norm is the
   convergence measure -- identical to ``l_bfgs_bounded``, so runs are
   directly comparable.
2. Approximately minimize the quadratic model
   ``m(s) = r_freeᵀ s + ½ sᵀ B s`` over the trust region
   ``||s||_M <= Delta``, restricted to the free subspace, with the
   Steihaug-Toint truncated CG (Nocedal & Wright, Algorithm 7.2; the
   preconditioned variant of Conn, Gould & Toint 2000, sec. 7.5.1). The
   model Hessian ``B`` is supplied matrix-free via ``hessp`` and only needs
   to be *bounded* -- inexact Hessian-vector products do not break global
   convergence; negative curvature is exploited by stepping to the
   trust-region boundary.
3. Accept or reject the projected trial point ``x_try = Pi_F(x + s)`` by the
   reduction ratio ``rho = (f - f_try) / pred`` and update the radius
   (reject below ``eta``; shrink x1/4 below rho=0.25; grow x2, capped, above
   rho=0.75 when the step reached the boundary). A boundary-limited step is
   never counted as converged.

**Preconditioning.** Passing ``precond`` (an application of ``M**-1`` for a
symmetric positive semi-definite ``M`` approximating ``B``) switches the
inner solve to preconditioned Steihaug and measures the trust region in the
``M``-norm, ``||s||_M**2 = sᵀ M s <= Delta**2``. This is equivalent to plain
Steihaug in the variable ``s_hat = M**(1/2) s``, where ``M**(-1/2) B
M**(-1/2)`` is well conditioned and a spherical region is the natural one.
Only ``M**-1`` is ever applied: the ``M``-inner products required by the
boundary test are propagated exactly by recursion, using ``M z = r`` for
``z = M**-1 r``,

    M d   = -r + beta * M d_prev        (seed  M d_0 = -r_0)
    M s  += alpha * M d                 (seed  M s_0 = 0)

so the cost is two extra stored vectors and no extra preconditioner
applications beyond the one ``z = M**-1 r`` per inner iteration.

Two consequences of preconditioning:

* ``delta0``/``delta_max`` are in **M-norm** units. For a stiffness-like
  ``M``, ``||s||_M ~ sqrt(2 * energy)``, i.e. units of sqrt(energy) rather
  than "typical change per variable" -- they must be recalibrated. The
  conversion factor is available for free as
  ``||z||_M / ||z||_2 = sqrt(zᵀg) / ||z||_2`` for ``z = M**-1 g``.
* The ``M d = -r + beta M d_prev`` recursion assumes ``z`` is *not* masked
  after ``M**-1`` is applied, so ``precond`` may only be combined with an
  unconstrained free set. This is checked and raises otherwise.

**Robustness to inexact (noisy) objectives.** Unlike a backtracking line
search -- whose sufficient-decrease test must resolve decreases that shrink
``∝ alpha`` towards zero and hence *always* drowns in evaluation noise
eventually -- the trust-region test compares the actual reduction against the
*computable* predicted reduction ``pred``. When the caller can bound its own
evaluation error, pass ``fun_error``/``request_accuracy``: the driver then
enforces the standard inexact-trust-region condition
``|f_err| <= eta_f * pred`` (Conn, Gould & Toint 2000, ch. 8.4/10.6; Kouri
et al., SISC 2013/14). With both hooks ``None`` this is a classic exact
trust-region method.

MPI-parallel: ``x``/gradient/bounds/``hessp``/``precond`` operate on *local*
per-rank slices; the scalar objective must be globally reduced; all
reductions go through the caller-supplied ``pnp``.
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

    ``sds``, ``dds``, ``sdd`` are the (already reduced) dot products
    ``sᵀGs``, ``dᵀGd``, ``sᵀGd`` for whichever metric ``G`` defines the
    region -- the identity for the unpreconditioned method, ``M`` for the
    preconditioned one. Returns ``(tau_minus, tau_plus)`` with
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


def _safety_factor(Delta, oldest):
    """Safety factor ``S`` of the adaptive delay (Meurant, Papez & Tichy,
    Numer. Algorithms 88, 2021; same as ``muFFTTO.solvers.findS``).

    With ``tail_j = sum_{i>=j} Delta_i`` (the current estimate of
    ``||s* - s_j||_B^2``): ``S = max_j tail_j / Delta_j`` over the iterations
    ``j`` from the last one whose estimate is still ``>= 1e4 tail_oldest`` (else
    from 0) up to the last but one. It guards against accepting an estimate
    during a phase of stagnation.
    """
    Delta = np.asarray(Delta)
    tail = np.cumsum(Delta[::-1])[::-1]
    far = np.nonzero(tail >= 1e4 * tail[oldest])[0]
    first = far[-1] if far.size > 0 else 0
    return float(np.max(tail[first:-1] / Delta[first:-1]))


def _steihaug_cg(hessp_free, g_free, delta, tol, maxiter, pnp, precond=None,
                 stop="energy", delay=10, delay_tau=None, metric=None):
    """Steihaug-Toint truncated CG on ``min m(s) = gᵀs + ½ sᵀBs`` s.t.
    ``||s||_M <= delta`` (Nocedal & Wright, Algorithm 7.2; preconditioned
    variant of Conn-Gould-Toint sec. 7.5.1).

    ``hessp_free`` must map a free-subspace vector to ``B v`` masked to the
    free subspace. ``precond`` (optional) must return ``M**-1 r`` for a
    symmetric positive semi-definite ``M``; ``None`` recovers the
    unpreconditioned algorithm exactly, with ``M = I``.

    ``metric`` (optional, with ``precond``) measures the trust region in a
    norm ``||s||_N`` other than the preconditioner's ``M``-norm: called as
    ``metric(z)`` right after ``z = precond(r)``, it returns ``N z``. The
    images ``N s``, ``N d`` follow from the CG recursions (``N d_{k+1} =
    -N z_{k+1} + beta N d_k``), so no extra operator application is needed
    when ``N z`` is a by-product of the preconditioner (e.g. the dual
    ``z = G K_d G r`` with ``N = G^-1``: ``N z = K_d G r``). Without
    ``metric``, ``N = M`` and ``N z = r``. With ``N != M`` the iterates'
    ``N``-norm need not grow monotonically, so the first boundary crossing
    is a heuristic stop.

    Returns ``(s, pred, on_boundary, status, nb_hessp)`` with
    ``pred = m(0) - m(s) = -m(s) >= 0`` the predicted reduction.

    Interior stopping test (``stop``):

    * ``"residual"``: ``||r_k|| <= tol ||g||`` (Euclidean).
    * ``"energy"`` (default): Hestenes-Stiefel lower bound of the energy-norm error
      with a fixed delay ``d = delay`` (Strakos & Tichy; Meurant, Papez &
      Tichy), made relative like a relative residual: with
      ``Delta_j = alpha_j (r_j, z_j) = ||s_{j+1} - s_j||_B^2``, stop at
      iteration ``k >= d`` once
      ``sum_{j=k-d}^{k} Delta_j <= tol**2 sum_{j=0}^{k} Delta_j``, i.e. the
      estimated error ``||s* - s_{k-d}||_B^2`` is below ``tol**2`` times the
      estimated ``||s* - s_0||_B^2``. The returned ``s_k`` is at least as
      accurate (the CG energy error decreases monotonically). Uses only CG
      scalars; no extra Hessian or preconditioner application.

      With ``delay_tau`` (e.g. 0.25) the delay is adaptive instead (Meurant,
      Papez & Tichy 2021, as in ``muFFTTO.solvers``): the estimate
      ``sum_{j=oldest}^{k} Delta_j`` of ``||s* - s_oldest||_B^2`` is accepted once
      ``S Delta_k / sum_{j=oldest}^{k-1} Delta_j <= delay_tau`` (``S``: see
      ``_safety_factor``) and ``k + 1 - oldest >= delay`` (``delay`` is then the
      minimal delay). CG stops when an accepted estimate is below ``tol**2``
      times the estimated ``||s* - s_0||_B^2``.

    Exits: interior tolerance (see ``stop``), negative curvature or
    trust-region crossing (both: step to the boundary), iteration cap, or
    -- with a semi-definite ``M`` -- exhaustion of the preconditioned
    residual. For negative curvature both sphere intersections are evaluated
    and the one with the smaller model value is taken (the muSpectre
    convention); for a trust-region crossing during a positive-curvature
    step the positive root is the minimizer along ``d``.
    """
    xp = array_namespace(g_free)
    s = xp.zeros_like(g_free)
    Bs = xp.zeros_like(g_free)  # running B s, for cheap model evaluations
    r = xp.asarray(g_free, copy=True)  # residual of grad m = g + B s at s = 0
    rr = pnp.sum(r * r)
    g_norm = math.sqrt(rr)
    if g_norm == 0.0:
        return s, 0.0, False, "zero gradient", 0
    # The stopping test stays on the *Euclidean* residual norm so that `tol`
    # keeps the same meaning as in the unpreconditioned method and remains
    # compatible with the outer forcing sequence.
    rtol2 = (tol * g_norm) ** 2
    if stop not in ("residual", "energy"):
        raise ValueError(f"stop must be 'residual' or 'energy', got {stop!r}")
    Delta = []  # Delta_j = alpha_j (r_j, z_j), for stop="energy"
    oldest = 0  # adaptive delay: oldest iterate whose error estimate is not accepted yet

    # z = M**-1 r, and the exactly-propagated M-images of s and d.
    z = xp.asarray(r, copy=True) if precond is None else precond(r)
    Ms = xp.zeros_like(g_free)  # M s (N s with `metric`), seed M s_0 = 0
    Md = -(r if metric is None else metric(z))  # M d_0 = M (-z_0) = -r_0

    d = -z
    rz = pnp.sum(r * z)
    if rz <= 0.0:
        # r lies in the null space of M**-1 (e.g. the pinned q=0 mode of a
        # Green operator): no descent direction available.
        return s, 0.0, False, "zero preconditioned gradient", 0

    def model(sv, Bsv):
        # m(s) = gᵀ s + ½ sᵀ B s, all-reduced.
        return pnp.sum(g_free * sv) + 0.5 * pnp.sum(sv * Bsv)

    nb_hessp = 0
    for _ in range(maxiter):
        Bd = hessp_free(d)
        nb_hessp += 1
        dBd = pnp.sum(d * Bd)

        # Metric inner products for the boundary test. With precond=None,
        # Ms == s and Md == d hold identically, so these reduce to the
        # Euclidean sᵀs, sᵀd, dᵀd of the plain algorithm.
        sMs = pnp.sum(s * Ms)
        sMd = pnp.sum(s * Md)
        dMd = pnp.sum(d * Md)

        if dMd <= 0.0:
            # M is degenerate along d: the M-norm region is not a sphere
            # here. Return the best feasible point found so far.
            return (s, max(-model(s, Bs), 0.0), False,
                    "degenerate metric", nb_hessp)

        if dBd <= 0.0:
            # Negative curvature: the model is unbounded along d; go to the
            # sphere. Evaluate both intersections, keep the smaller model.
            tau_m, tau_p = _to_boundary(s, d, sMs, dMd, sMd, delta)
            cand = []
            for tau in (tau_m, tau_p):
                sv = s + tau * d
                Bsv = Bs + tau * Bd
                cand.append((model(sv, Bsv), sv, Bsv))
            m_val, s, Bs = min(cand, key=lambda c: c[0])
            return s, -m_val, True, "negative curvature", nb_hessp

        alpha = rz / dBd
        # Would the CG step leave the trust region?
        step_norm2 = sMs + 2.0 * alpha * sMd + alpha * alpha * dMd
        if step_norm2 >= delta * delta:
            _, tau = _to_boundary(s, d, sMs, dMd, sMd, delta)
            s = s + tau * d
            Bs = Bs + tau * Bd
            return s, -model(s, Bs), True, "boundary", nb_hessp

        s = s + alpha * d
        Bs = Bs + alpha * Bd
        Ms = Ms + alpha * Md
        r = r + alpha * Bd

        new_rr = pnp.sum(r * r)
        if stop == "residual":
            if new_rr <= rtol2:
                return s, -model(s, Bs), False, "tolerance", nb_hessp
        else:
            # rz still holds (r_k, z_k): Delta_k = alpha_k (r_k, z_k) (global scalars)
            Delta.append(alpha * rz)
            k = len(Delta) - 1
            if delay_tau is None:  # fixed delay
                if k >= delay and (sum(Delta[k - delay:])
                                   <= tol * tol * sum(Delta)):
                    return s, -model(s, Bs), False, "tolerance", nb_hessp
            elif k >= 1:  # adaptive delay, at least `delay`
                S = _safety_factor(Delta, oldest)
                while len(Delta) - oldest >= delay and S * Delta[k] <= delay_tau * sum(Delta[oldest:k]):
                    if sum(Delta[oldest:]) <= tol * tol * sum(Delta):
                        return s, -model(s, Bs), False, "tolerance", nb_hessp
                    oldest += 1

        z = xp.asarray(r, copy=True) if precond is None else precond(r)
        Mz = r if metric is None else metric(z)
        new_rz = pnp.sum(r * z)
        if new_rz <= 0.0:
            return (s, -model(s, Bs), False,
                    "preconditioned residual exhausted", nb_hessp)

        beta = new_rz / rz
        d = -z + beta * d
        Md = -Mz + beta * Md  # exact: M z = r by construction (N z from `metric`)
        rz = new_rz
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
    precond=None,
    precond_metric=None,
    gtol=1e-5,
    maxiter=200,
    delta0=0.05,
    delta_max=0.5,
    eta=0.1,
    inner_tol=None,
    inner_stop="energy",
    inner_delay=10,
    inner_tau=None,
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
    Bound-constrained trust-region Newton-CG with optional index pinning,
    optional preconditioning and optional inexact-evaluation control.

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
        Extra positional arguments forwarded to ``fun`` / ``jac`` / ``hessp``
        / ``precond``.
    jac : bool or callable, optional
        See ``fun``. Default ``None`` means ``jac=True`` (fused).
    bounds_lo, bounds_hi : array_like or scalar, optional
        Box bounds. ``None`` means unbounded on that side.
    zero_mask : array_like of bool, optional
        Indices pinned to ``x = 0`` (e.g. contact nodes).
    precond : callable, optional
        ``precond(r, *args)`` returns ``M**-1 r`` for a symmetric positive
        semi-definite ``M`` approximating ``B`` (local slices in and out).
        Switches the inner solve to preconditioned Steihaug and measures the
        trust region in the ``M``-norm -- see the module docstring;
        ``delta0``/``delta_max`` then carry ``M``-norm units and must be
        recalibrated. Only supported with an unconstrained free set
        (``bounds_lo``, ``bounds_hi`` and ``zero_mask`` all ``None``),
        because the internal ``M d = -r + beta M d_prev`` recursion requires
        that ``z`` is not masked after ``M**-1`` is applied.
    precond_metric : callable, optional
        With ``precond``: measure the trust region in a norm ``||s||_N``
        instead of the ``M``-norm. ``precond_metric(z)``, called right after
        ``z = precond(r)``, returns ``N z`` (local slices); see
        ``_steihaug_cg``.
    gtol : float, optional
        Convergence tolerance on the infinity norm of the box-masked
        gradient (same measure as ``l_bfgs_bounded``). Default 1e-5.
    maxiter : int, optional
        Maximum number of outer iterations. Default 200.
    delta0, delta_max : float, optional
        Initial and maximum trust-region radius in **RMS-per-component
        units**: the radius is ``delta * sqrt(N_global)``, so the same value
        expresses "typical change per variable" at any problem size. With
        ``precond`` the radius is an ``M``-norm and the interpretation
        changes accordingly. Defaults 0.05 and 0.5.
    eta : float, optional
        Step acceptance threshold: accept iff ``rho >= eta``. Default 0.1.
    inner_tol : float, optional
        Relative tolerance of the Steihaug CG (see ``inner_stop``).
        ``None`` (default) uses the superlinear forcing sequence
        ``min(0.5, sqrt(||g_free|| / ||g_free,0||))`` (Nocedal & Wright,
        p. 169), with the norms relative to that at the first iterate so
        that the sequence does not depend on the units of the objective.
    inner_stop : str, optional
        Interior stopping test of the Steihaug CG: ``"energy"`` (default;
        Hestenes-Stiefel lower bound of the energy error with delay
        ``inner_delay``, relative to the estimated initial energy error; see
        ``_steihaug_cg``) or ``"residual"`` (Euclidean residual relative
        to the gradient, the original test). ``inner_tol`` / the forcing sequence is the
        relative tolerance in both cases (on the energy *norm* for
        ``"energy"``).
    inner_delay : int, optional
        Delay ``d`` of the energy-error lower bound (``inner_stop="energy"``);
        at least ``d`` CG iterations are done unless another exit is taken.
        Default 10.
    inner_tau : float, optional
        ``None`` (default): the delay is fixed to ``inner_delay``. A value
        (e.g. 0.25): adaptive delay (Meurant, Papez & Tichy 2021) with
        tolerance ``inner_tau`` and minimal delay ``inner_delay``; see
        ``_steihaug_cg``.
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
        the most recent ``fun`` evaluation. Enables the inexact-trust-region
        accuracy control.
    request_accuracy : callable, optional
        ``request_accuracy(target)`` asks the caller to tighten its inner
        solves until the ``fun`` evaluation error is below the absolute
        value ``target``.
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
        ``nb_precond`` (total preconditioner applications),
        ``delta_history``, ``rho_history`` (per outer iteration; ``rho`` is
        ``nan`` where the model predicted no reduction).

    Notes
    -----
    **MPI contract.** As for ``l_bfgs_bounded``: ``x``, the gradient, the
    bounds, ``zero_mask`` and the vectors seen by ``hessp`` and ``precond``
    are all *local* per-rank slices of the global problem; the scalar energy
    returned by ``fun`` (and the error bound returned by ``fun_error``) must
    be **globally reduced** (identical on every rank).
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

    if precond is not None and not (
        bounds_lo is None and bounds_hi is None and zero_mask is None
    ):
        raise NotImplementedError(
            "`precond` requires an unconstrained free set: the internal "
            "M d = -r + beta M d_prev recursion assumes z = M**-1 r is not "
            "masked to the free subspace after the preconditioner is "
            "applied. Drop the bounds, or supply a preconditioner that "
            "commutes with the active-set mask and adapt _steihaug_cg."
        )

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

    nb_precond_total = 0

    def _precond_flat(v):
        nonlocal nb_precond_total
        nb_precond_total += 1
        return _flat_result(precond(_shaped(v), *args))

    precond_flat = None if precond is None else _precond_flat
    metric_flat = (None if precond_metric is None else
                   lambda z: _flat_result(precond_metric(_shaped(z))))

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
    # absolute radii (Euclidean, or M-norm with precond) and caps the inner CG.
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
                "nb_precond": int(nb_precond_total),
                "delta_history": delta_history,
                "rho_history": rho_history,
            }
        )

    if disp and (comm is None or pnp is np or comm.rank == 0):
        print(
            f"{'iter':<5} {'f':<14} {'|r|inf_free':<14} {'Delta_rms':<10} "
            f"{'rho':<10} {'m_cg':<5} {'step':<6} {'cg_exit':<28}"
        )
        print("-" * 96)

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
            hessp_free, r_free, delta, tol_k, inner_maxiter, red,
            precond=precond_flat, stop=inner_stop, delay=inner_delay,
            delay_tau=inner_tau, metric=metric_flat,
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
        # accuracy floor -- stop retrying and let the rho-test decide with
        # the best available accuracy.
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
                f"{nb_hessp:<5d} {'acc' if accepted else 'rej':<6} "
                f"{status:<28}"
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
