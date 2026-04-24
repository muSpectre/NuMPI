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
Projected L-BFGS for problems with a single linear equality constraint and
optional box bounds.

Feasible set:

    F = { x : <a, x> = target, lo <= x <= hi, x[zero_mask] = 0 }

The multiplier ``lambda = <a, grad>/<a, a>`` is available in closed form
(the constraint is linear), so the problem is solved as a single-loop
projected-gradient quasi-Newton rather than an augmented-Lagrangian outer
loop. Each iteration:

1. Tangent-project the gradient:  ``Gp = grad - lambda * a``.
2. Build a two-loop L-BFGS direction from ``(s, y)`` pairs *of tangent
   gradients*, then re-tangent-project against numerical drift.
3. Backtracking Armijo line search with the projection arc
   ``x_try = Pi_F(x + alpha * d)``.

Convergence is measured by the infinity-norm of the box-active-masked
tangent gradient ("free KKT residual"), which is scale-invariant in N.

MPI-parallel: all reductions go through the caller-supplied ``pnp``.
"""

import logging

import numpy as np

from ..Tools import Reduction
from .Result import OptimizeResult

_log = logging.getLogger(__name__)


def _twoloop_direction(g, s_hist, y_hist, rho_hist, pnp):
    """
    Two-loop L-BFGS recursion: returns ``-H * g`` where ``H`` is the L-BFGS
    Hessian approximation with Nocedal-Wright's ``gamma = <s, y>/<y, y>``
    initial scaling on the newest pair (identity if the history is empty).

    Histories are most-recent-first lists.
    """
    q = g.copy()
    n = len(s_hist)
    a = np.empty(n)
    for k in range(n):  # newest -> oldest
        a[k] = rho_hist[k] * pnp.sum(s_hist[k] * q)
        q -= a[k] * y_hist[k]
    if n > 0:
        gamma = pnp.sum(s_hist[0] * y_hist[0]) / pnp.sum(y_hist[0] * y_hist[0])
        q *= gamma
    for k in range(n - 1, -1, -1):  # oldest -> newest
        beta = rho_hist[k] * pnp.sum(y_hist[k] * q)
        q += (a[k] - beta) * s_hist[k]
    return -q


def _kkt_residual(Gp, x, lo, hi, zero_mask=None, tol_box=1e-12):
    """
    Box-active-masked tangent gradient. Components where ``x`` sits on an
    active box face *and* ``Gp`` points into that face are zeroed
    (complementary slackness satisfied, no improvement possible). Contact
    nodes are always zeroed.
    """
    r = Gp.copy()
    if zero_mask is not None:
        r = np.where(zero_mask, 0.0, r)
    if lo is not None:
        r = np.where((x <= lo + tol_box) & (Gp >= 0.0), 0.0, r)
    if hi is not None:
        r = np.where((x >= hi - tol_box) & (Gp <= 0.0), 0.0, r)
    return r


def _free_mask(x, lo, hi, zero_mask=None, tol_box=1e-12):
    """
    Boolean mask marking nodes that are *free* — interior to the box and not
    in the contact set. The multiplier for the equality constraint must be
    taken over this mask (the active-bound indices contribute their own
    multipliers to the KKT system and should not be lumped in).
    """
    free = np.ones_like(x, dtype=bool)
    if zero_mask is not None:
        free &= ~zero_mask
    if lo is not None:
        free &= x > np.asarray(lo) + tol_box
    if hi is not None:
        free &= x < np.asarray(hi) - tol_box
    return free


def _tangent_gradient(grad, x, linear_constraint, lo, hi, zero_mask):
    """
    Project the full gradient onto the tangent of the equality constraint
    *restricted to the free indices*. Returns ``(Gp, lam)`` where Gp is the
    tangent gradient (zeroed at contact nodes) and lam is the multiplier.
    """
    g_flat = np.asarray(grad).ravel()
    x_flat = np.asarray(x).ravel()
    mask = _free_mask(
        x_flat,
        lo,
        hi,
        zero_mask=(None if zero_mask is None else np.asarray(zero_mask).ravel()),
    )
    lam = linear_constraint.multiplier(g_flat, mask=mask)
    Gp = g_flat - lam * linear_constraint.a
    if zero_mask is not None:
        Gp = np.where(np.asarray(zero_mask).ravel(), 0.0, Gp)
    return Gp.reshape(np.asarray(grad).shape), lam


def l_bfgs_projected(
    fun,
    x0,
    linear_constraint,
    args=(),
    jac=None,
    bounds_lo=None,
    bounds_hi=None,
    zero_mask=None,
    gtol=1e-5,
    maxiter=500,
    maxcor=10,
    c1=1e-4,
    max_halvings=40,
    comm=None,
    pnp=None,
    callback=None,
    disp=False,
):
    """
    Projected L-BFGS for a problem with one linear equality constraint and
    optional box bounds.

    Parameters
    ----------
    fun : callable
        Objective. If ``jac`` is ``True``, ``fun(x, *args)`` must return
        ``(energy, gradient)``. If ``jac`` is a callable, ``fun`` returns
        only the energy and ``jac(x, *args)`` returns the gradient.
    x0 : ndarray
        Initial guess. Will be projected onto ``F`` before the loop starts.
    linear_constraint : LinearConstraint
        The equality constraint ``<a, x> = target``.
    args : tuple, optional
        Extra positional arguments forwarded to ``fun`` / ``jac``.
    jac : bool or callable, optional
        See ``fun``. Default ``None`` means ``jac=True`` (fused).
    bounds_lo, bounds_hi : array_like or scalar, optional
        Box bounds. ``None`` means unbounded on that side.
    zero_mask : array_like of bool, optional
        Indices pinned to ``x = 0`` (contact nodes).
    gtol : float, optional
        Convergence tolerance on the infinity norm of the box-masked tangent
        gradient. Default 1e-5.
    maxiter : int, optional
        Maximum number of outer iterations. Default 500.
    maxcor : int, optional
        L-BFGS memory (number of ``(s, y)`` pairs retained). Default 10.
    c1 : float, optional
        Armijo sufficient-decrease coefficient. Default 1e-4.
    max_halvings : int, optional
        Maximum step halvings per line search. Default 40.
    comm, pnp : MPI.Comm or Reduction, optional
        MPI communicator or pre-built reduction wrapper. Pass one, not both.
    callback : callable, optional
        ``callback(x)`` called after each iteration.
    disp : bool, optional
        Print a per-iteration diagnostic table.

    Returns
    -------
    OptimizeResult

    Notes
    -----
    **MPI contract.** ``x``, the gradient, ``bounds_lo``, ``bounds_hi``,
    ``zero_mask`` and ``linear_constraint.a`` are all *local* per-rank slices
    of the global problem; the scalar energy returned by ``fun`` and
    ``linear_constraint.target`` are **global** (same on every rank). The
    scalar energy in particular must be globally reduced before being
    returned — otherwise ranks disagree in the Armijo acceptance test and
    the iterates diverge silently across ranks. See the "MPI Conventions"
    section of the top-level README for details and a worked example.
    """
    if comm is not None:
        if pnp is not None:
            raise RuntimeError("Please specify either `comm` or `pnp`, not both.")
        pnp = Reduction(comm)
    elif pnp is None:
        pnp = np

    if jac is True or jac is None:

        def fun_grad(x):
            return fun(x, *args)

    elif jac is False:
        raise NotImplementedError("Numerical evaluation of gradient not implemented")
    else:

        def fun_grad(x):
            return fun(x, *args), jac(x, *args)

    if linear_constraint is None:
        raise ValueError("l_bfgs_projected requires a LinearConstraint.")

    original_shape = np.asarray(x0).shape

    def project(y):
        return linear_constraint.project(
            y, lo=bounds_lo, hi=bounds_hi, zero_mask=zero_mask
        )

    # --- Feasible starting iterate and initial tangent gradient ----------
    x = project(x0.ravel()).reshape(original_shape)
    phi, grad = fun_grad(x)
    Gp, lam = _tangent_gradient(
        grad, x, linear_constraint, bounds_lo, bounds_hi, zero_mask
    )

    r_free = _kkt_residual(Gp, x, lo=bounds_lo, hi=bounds_hi, zero_mask=zero_mask)
    r_norm = pnp.max(np.abs(r_free))

    # L-BFGS and the Armijo test operate on the KKT-masked gradient `r_free`,
    # not on the raw tangent gradient `Gp`. Box-active nodes with the "right
    # sign" gradient (pointing further into the bound) can't move; including
    # their contribution in <gradient, d> overstates the expected decrease
    # and causes the line search to reject steps that are in fact optimal.
    s_hist, y_hist, rho_hist = [], [], []

    if disp and (comm is None or pnp is np or comm.rank == 0):
        print(f"{'iter':<5} {'f':<14} {'|Gp|inf_free':<14} {'lambda':<14} {'alpha':<8}")
        print("-" * 64)
        print(f"{0:<5d} {phi:<14.6e} {r_norm:<14.4e} {lam:<14.4e} {'-':<8}")

    if r_norm < gtol:
        return _result(
            True,
            x,
            phi,
            grad,
            lam,
            0,
            r_norm,
            "CONVERGENCE: NORM_OF_PROJECTED_GRADIENT_<=_GTOL",
        )

    for iteration in range(1, maxiter + 1):
        # L-BFGS search direction from the KKT-masked gradient; re-tangent-
        # project against numerical drift.
        d = _twoloop_direction(r_free.ravel(), s_hist, y_hist, rho_hist, pnp)
        mu = pnp.sum(linear_constraint.a * d) / linear_constraint.aTa
        d = d - mu * linear_constraint.a
        if zero_mask is not None:
            d = np.where(np.asarray(zero_mask).ravel(), 0.0, d)
        d = d.reshape(original_shape)

        gd = pnp.sum(r_free * d)
        if gd >= 0.0:
            # Stale curvature produced a non-descent direction: reset.
            s_hist.clear()
            y_hist.clear()
            rho_hist.clear()
            d = -r_free.copy()
            gd = pnp.sum(r_free * d)

        # Projected Armijo backtracking. If the L-BFGS direction fails the
        # line search (typically because the active set just flipped), fall
        # back to scaled steepest descent once before bailing out.
        step = _armijo(
            fun_grad,
            x,
            d,
            phi,
            gd,
            project,
            c1=c1,
            max_halvings=max_halvings,
            alpha_init=1.0,
        )
        if step is None:
            s_hist.clear()
            y_hist.clear()
            rho_hist.clear()
            d = -r_free.copy()
            gd = pnp.sum(r_free * d)
            g_norm = np.sqrt(pnp.sum(r_free * r_free))
            alpha_init = 1.0 / g_norm if g_norm > 0 else 1.0
            step = _armijo(
                fun_grad,
                x,
                d,
                phi,
                gd,
                project,
                c1=c1,
                max_halvings=max_halvings,
                alpha_init=alpha_init,
            )
        if step is None:
            _log.info("line-search did not converge")
            return _result(
                False,
                x,
                phi,
                grad,
                lam,
                iteration,
                r_norm,
                "CONVERGENCE: line-search did not converge",
            )
        x_new, phi_new, grad_new, alpha = step

        Gp_new, lam_new = _tangent_gradient(
            grad_new, x_new, linear_constraint, bounds_lo, bounds_hi, zero_mask
        )
        r_free_new = _kkt_residual(
            Gp_new, x_new, lo=bounds_lo, hi=bounds_hi, zero_mask=zero_mask
        )

        # L-BFGS curvature pair uses the KKT-masked gradient so the Hessian
        # approximation tracks the "free" subspace on which the optimiser
        # actually moves, not components pinned by the box.
        s = x_new - x
        y = r_free_new - r_free
        sy = pnp.sum(s * y)
        s_norm = np.sqrt(pnp.sum(s * s))
        y_norm = np.sqrt(pnp.sum(y * y))
        if sy > 1e-12 * (s_norm * y_norm + np.finfo(float).eps):
            s_hist.insert(0, s.copy())
            y_hist.insert(0, y.copy())
            rho_hist.insert(0, 1.0 / sy)
            if len(s_hist) > maxcor:
                s_hist.pop()
                y_hist.pop()
                rho_hist.pop()

        x, phi, grad, Gp, lam, r_free = (
            x_new,
            phi_new,
            grad_new,
            Gp_new,
            lam_new,
            r_free_new,
        )
        r_norm = pnp.max(np.abs(r_free))

        if disp and (comm is None or pnp is np or comm.rank == 0):
            print(
                f"{iteration:<5d} {phi:<14.6e} {r_norm:<14.4e} {lam:<14.4e} {alpha:<8.2e}"
            )

        if callback is not None:
            callback(x)

        if r_norm < gtol:
            return _result(
                True,
                x,
                phi,
                grad,
                lam,
                iteration,
                r_norm,
                "CONVERGENCE: NORM_OF_PROJECTED_GRADIENT_<=_GTOL",
            )

    return _result(
        False,
        x,
        phi,
        grad,
        lam,
        maxiter,
        r_norm,
        "NO CONVERGENCE: MAXITERATIONS REACHED",
    )


def _armijo(fun_grad, x, d, phi, gd, project, *, c1, max_halvings, alpha_init):
    """
    Projected backtracking line search. Returns ``(x_new, phi_new, grad_new, alpha)``
    on success, ``None`` on failure.
    """
    alpha = alpha_init
    for _ in range(max_halvings):
        x_try = project(x + alpha * d)
        phi_try, grad_try = fun_grad(x_try)
        if phi_try <= phi + c1 * alpha * gd:
            return x_try, phi_try, grad_try, alpha
        alpha *= 0.5
    return None


def _result(success, x, phi, grad, lam, nit, residual, message):
    return OptimizeResult(
        {
            "success": bool(success),
            "x": x,
            "fun": phi,
            "jac": grad,
            "multiplier": lam,
            "nit": int(nit),
            "message": message,
            "max_grad": float(residual),
        }
    )
