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
Bound-constrained L-BFGS with optional index pinning.

Feasible set:

    F = { x : lo <= x <= hi, x[zero_mask] = 0 }

Each iteration:

1. Build the box-active-masked gradient ``r_free`` (components where ``x``
   sits on an active bound *and* the gradient points further into the bound
   are zeroed; contact nodes are always zeroed).
2. Form a two-loop L-BFGS direction from ``(s, y)`` pairs *of masked
   gradients* — ties the Hessian approximation to the free subspace, which
   is the only place the iterate can actually move.
3. Backtracking Armijo line search with the projection arc
   ``x_try = Pi_F(x + alpha * d)``, where ``Pi_F`` is elementwise clipping.

Convergence is measured by the infinity-norm of ``r_free`` ("free KKT
residual"), which is scale-invariant in N.

MPI-parallel: all reductions go through the caller-supplied ``pnp``.
"""

import logging

import numpy as np

from ..Tools import Reduction
from ._lbfgs_helpers import _armijo, _kkt_residual, _twoloop_direction
from .Result import OptimizeResult

_log = logging.getLogger(__name__)


def l_bfgs_bounded(
    fun,
    x0,
    args=(),
    jac=None,
    bounds_lo=None,
    bounds_hi=None,
    zero_mask=None,
    gtol=1e-5,
    ftol=0.0,
    xtol=0.0,
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
    Bound-constrained L-BFGS with optional index pinning.

    Parameters
    ----------
    fun : callable
        Objective. If ``jac`` is ``True`` (or ``None``), ``fun(x, *args)``
        must return ``(energy, gradient)``. If ``jac`` is a callable,
        ``fun`` returns only the energy and ``jac(x, *args)`` returns the
        gradient.
    x0 : ndarray
        Initial guess. Will be projected onto ``F`` before the loop starts.
    args : tuple, optional
        Extra positional arguments forwarded to ``fun`` / ``jac``.
    jac : bool or callable, optional
        See ``fun``. Default ``None`` means ``jac=True`` (fused).
    bounds_lo, bounds_hi : array_like or scalar, optional
        Box bounds. ``None`` means unbounded on that side.
    zero_mask : array_like of bool, optional
        Indices pinned to ``x = 0`` (e.g. contact nodes).
    gtol : float, optional
        Convergence tolerance on the infinity norm of the box-masked
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
    **MPI contract.** ``x``, the gradient, ``bounds_lo``, ``bounds_hi`` and
    ``zero_mask`` are all *local* per-rank slices of the global problem; the
    scalar energy returned by ``fun`` is **global** (same on every rank).
    The scalar energy in particular must be globally reduced before being
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

    original_shape = np.asarray(x0).shape

    # Work internally on the flattened (1D) iterate so that the iterate, the
    # gradient and the L-BFGS (s, y) history all share a single shape; the
    # two-loop recursion then never mixes raveled and n-D arrays. The user's
    # ``fun``/``jac`` and ``callback`` still see the original n-D shape, and the
    # result is reshaped back to it. ``x``, the gradient, the bounds and the
    # zero mask are all handled per-rank; flattening is purely local.
    def _flat(a):
        return a if (a is None or np.isscalar(a)) else np.asarray(a).ravel()

    bounds_lo = _flat(bounds_lo)
    bounds_hi = _flat(bounds_hi)
    zero_mask = (
        None if zero_mask is None else np.asarray(zero_mask).ravel().astype(bool)
    )

    if jac is True or jac is None:

        def fun_grad(x):
            phi, g = fun(x.reshape(original_shape), *args)
            return phi, np.asarray(g).ravel()

    elif jac is False:
        raise NotImplementedError("Numerical evaluation of gradient not implemented")
    else:

        def fun_grad(x):
            xr = x.reshape(original_shape)
            return fun(xr, *args), np.asarray(jac(xr, *args)).ravel()

    def project(y):
        if bounds_lo is None and bounds_hi is None:
            z = np.asarray(y, dtype=float)
        else:
            z = np.clip(y, bounds_lo, bounds_hi)
        if zero_mask is not None:
            z = np.where(zero_mask, 0.0, z)
        return z

    def _finish(success, x, phi, grad, nit, residual, message):
        # Return the iterate and gradient in the caller's original shape.
        return _result(
            success, np.asarray(x).reshape(original_shape), phi,
            np.asarray(grad).reshape(original_shape), nit, residual, message,
        )

    # --- Feasible starting iterate --------------------------------------
    x = project(np.asarray(x0, dtype=float).ravel())
    phi, grad = fun_grad(x)

    r_free = _kkt_residual(
        grad, x, lo=bounds_lo, hi=bounds_hi, zero_mask=zero_mask
    )
    r_norm = pnp.max(np.abs(r_free))

    # L-BFGS and the Armijo test operate on the KKT-masked gradient `r_free`,
    # not on the raw gradient. Box-active nodes with the "right sign" gradient
    # (pointing further into the bound) can't move; including their
    # contribution in <gradient, d> overstates the expected decrease and
    # causes the line search to reject steps that are in fact optimal.
    s_hist, y_hist, rho_hist = [], [], []

    if disp and (comm is None or pnp is np or comm.rank == 0):
        print(f"{'iter':<5} {'f':<14} {'|r|inf_free':<14} {'alpha':<8}")
        print("-" * 46)
        print(f"{0:<5d} {phi:<14.6e} {r_norm:<14.4e} {'-':<8}")

    if r_norm < gtol:
        return _finish(
            True, x, phi, grad, 0, r_norm,
            "CONVERGENCE: NORM_OF_PROJECTED_GRADIENT_<=_GTOL",
        )

    for iteration in range(1, maxiter + 1):
        # L-BFGS search direction from the KKT-masked gradient (all 1D).
        d = _twoloop_direction(r_free, s_hist, y_hist, rho_hist, pnp)
        if zero_mask is not None:
            d = np.where(zero_mask, 0.0, d)

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
            fun_grad, x, d, phi, gd, project,
            c1=c1, max_halvings=max_halvings, alpha_init=1.0,
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
                fun_grad, x, d, phi, gd, project,
                c1=c1, max_halvings=max_halvings, alpha_init=alpha_init,
            )
        if step is None:
            _log.info("line-search did not converge")
            return _finish(
                False, x, phi, grad, iteration, r_norm,
                "CONVERGENCE: line-search did not converge",
            )
        x_new, phi_new, grad_new, alpha = step

        r_free_new = _kkt_residual(
            np.asarray(grad_new), x_new,
            lo=bounds_lo, hi=bounds_hi, zero_mask=zero_mask,
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

        if xtol > 0:
            if pnp.max(np.abs(x_new - x)) < xtol:
                return _finish(
                    True,
                    x,
                    phi,
                    grad,
                    iteration,
                    r_norm,
                    "CONVERGENCE: NORM_OF_VARIABLE_STEP_<=_XTOL",
                )

        if ftol > 0:
            if abs(phi_new - phi) < ftol * max(1, abs(phi_new), abs(phi)):
                return _finish(
                    True,
                    x,
                    phi,
                    grad,
                    iteration,
                    r_norm,
                    "CONVERGENCE: REL_REDUCTION_OF_F_<=_FTOL",
                )

        x, phi, grad, r_free = x_new, phi_new, grad_new, r_free_new
        r_norm = pnp.max(np.abs(r_free))

        if disp and (comm is None or pnp is np or comm.rank == 0):
            print(f"{iteration:<5d} {phi:<14.6e} {r_norm:<14.4e} {alpha:<8.2e}")

        if callback is not None:
            callback(x.reshape(original_shape))

        if r_norm < gtol:
            return _finish(
                True, x, phi, grad, iteration, r_norm,
                "CONVERGENCE: NORM_OF_PROJECTED_GRADIENT_<=_GTOL",
            )

    return _finish(
        False, x, phi, grad, maxiter, r_norm,
        "NO CONVERGENCE: MAXITERATIONS REACHED",
    )


def _result(success, x, phi, grad, nit, residual, message):
    return OptimizeResult(
        {
            "success": bool(success),
            "x": x,
            "fun": phi,
            "jac": grad,
            "nit": int(nit),
            "message": message,
            "max_grad": float(residual),
        }
    )
