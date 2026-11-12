#
# Copyright 2026 Yizhen Wang
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
Projected L-BFGS for problems with quadratic equality constraints and optional
box bounds.

Feasible set:

    F = { x : x^T B_j x / 2 + <a_j, x> = target_j for all j,
              lo <= x <= hi, x[zero_mask] = 0 }

The constraints are one ``QuadraticConstraint``, which provides the
multipliers, the tangent projection and the restoration of trial points.

MPI-parallel: all reductions go through the caller-supplied ``pnp``.
"""

import logging

import numpy as np

from ..Tools import Reduction
from ..Tools.Reduction import same_reduction_group
from ._lbfgs_helpers import _twoloop_direction
from .QuadraticConstraint import _active
from .Result import OptimizeResult

_log = logging.getLogger(__name__)

# The products B_j x are carried from one iterate to the next by linearity;
# they are recomputed after this many iterations to bound the rounding error.
_REFRESH_PRODUCTS = 20


def l_bfgs_qc_projected(
    fun,
    x0,
    quadratic_constraint,
    args=(),
    jac=None,
    bounds_lo=None,
    bounds_hi=None,
    zero_mask=None,
    scale=None,
    gtol=1e-5,
    ftol=0.0,
    xtol=0.0,
    maxiter=500,
    maxcor=10,
    c1=1e-4,
    max_halvings=40,
    max_restore_rounds=20,
    comm=None,
    pnp=None,
    callback=None,
    disp=False,
):
    """
    Projected L-BFGS for a problem with quadratic equality constraints and
    optional box bounds.

    Args:
        fun: Objective. If ``jac`` is ``True`` or ``None``, ``fun(x, *args)``
            returns ``(energy, gradient)``. If ``jac`` is a callable, ``fun``
            returns the energy and ``jac(x, *args)`` the gradient. The energy
            is globally reduced.
        x0: Initial guess, restored onto the feasible set before the first
            iteration.
        quadratic_constraint: The equality constraints, a
            ``QuadraticConstraint`` on the flattened ``x``.
        args: Extra positional arguments forwarded to ``fun`` and ``jac``.
        jac: See ``fun``.
        bounds_lo, bounds_hi: Box bounds, scalar or array. ``None`` means
            unbounded on that side.
        zero_mask: Boolean array of the entries pinned to zero.
        scale: Positive scale ``s`` of the unknowns, scalar or array shaped
            like ``x0``. The solver iterates on ``u = x / s``: the tangent
            projection, the restoration, the L-BFGS update, ``gtol`` and
            ``xtol`` all refer to ``u``. Choose ``s`` as the typical
            magnitude of each unknown. ``None`` means 1.
        gtol: Tolerance on the infinity norm of the reduced gradient with the
            held entries zeroed (with ``scale``, of ``s`` times that
            gradient).
        ftol: Tolerance on the relative reduction of the energy. 0 disables it.
        xtol: Tolerance on the infinity norm of the step. 0 disables it.
        maxiter: Maximum number of iterations.
        maxcor: Number of L-BFGS curvature pairs kept.
        c1: Armijo sufficient-decrease coefficient.
        max_halvings: Maximum step halvings per line search.
        max_restore_rounds: Maximum restorations per trial point.
        comm, pnp: MPI communicator or reduction wrapper. Pass one, not both.
        callback: ``callback(x)`` called after each iteration.
        disp: Print a per-iteration table.

    Returns:
        ``OptimizeResult`` with ``x``, ``fun``, ``jac``, ``multiplier`` (one
        entry per constraint component), ``nit``, ``success``, ``message``,
        ``max_grad`` and ``constr_violation`` (the largest absolute residual
        of the constraint components at ``x``).

    Raises:
        RuntimeError: ``comm`` and ``pnp`` are both given, or ``x0`` cannot be
            restored onto the feasible set.
        ValueError: The constraint is missing, its reduction does not span
            the same ranks as the solver's, its ``hessp`` is not a symmetric
            linear map (``QuadraticConstraint.check_hessp``), or ``scale`` is
            not positive and finite.
    """
    if comm is not None:
        if pnp is not None:
            raise RuntimeError("Please specify either `comm` or `pnp`, not both.")
        pnp = Reduction(comm)
    elif pnp is None:
        pnp = np

    qc = quadratic_constraint
    if qc is None:
        raise ValueError("l_bfgs_qc_projected requires a QuadraticConstraint.")
    if not same_reduction_group(qc.pnp, pnp):
        raise ValueError(
            "The QuadraticConstraint's reduction does not span the same MPI "
            "ranks as l_bfgs_qc_projected."
        )
    # The multipliers, the tangent projection and the restoration are exact
    # only for a symmetric linear hessp; anything else would be solved
    # silently as a different problem.
    qc.check_hessp()

    original_shape = np.asarray(x0).shape

    def _flat(v):
        return v if (v is None or np.isscalar(v)) else np.asarray(v).ravel()

    bounds_lo = _flat(bounds_lo)
    bounds_hi = _flat(bounds_hi)
    zero_mask = (
        None if zero_mask is None else np.asarray(zero_mask).ravel().astype(bool)
    )

    # With a scale s the iteration runs on u = x / s: the objective gradient
    # becomes s * grad, the bounds lo / s and hi / s, and the constraint the
    # one in u. Below, `x` and `grad` are these scaled quantities.
    if scale is None:
        s = 1.0
    else:
        s = np.broadcast_to(
            np.asarray(scale, dtype=float), original_shape
        ).ravel().copy()
        if not (np.all(np.isfinite(s)) and np.all(s > 0.0)):
            raise ValueError("l_bfgs_qc_projected: scale must be positive and finite")
        qc = qc.scaled(s)
        bounds_lo = None if bounds_lo is None else bounds_lo / s
        bounds_hi = None if bounds_hi is None else bounds_hi / s

    if jac is True or jac is None:

        def fun_grad(x):
            phi, g = fun((s * x).reshape(original_shape), *args)
            return phi, s * np.asarray(g).ravel()

    elif jac is False:
        raise NotImplementedError("Numerical evaluation of gradient not implemented")
    else:

        def fun_grad(x):
            xr = (s * x).reshape(original_shape)
            return fun(xr, *args), s * np.asarray(jac(xr, *args)).ravel()

    def restore(y, products=None, fixed=None):
        return qc.restore(
            y, lo=bounds_lo, hi=bounds_hi, zero_mask=zero_mask,
            max_rounds=max_restore_rounds, products=products, fixed=fixed,
        )

    def state(x, grad, N, G_prev=None):
        # Held entries, multipliers, reduced gradient and its masked version at
        # x with normals N; with G_prev, also P(x) G_prev, from the same
        # reduction.
        at_lo, at_hi, zero = _active(x, bounds_lo, bounds_hi, zero_mask)
        free = ~(at_lo | at_hi | zero)
        vectors = grad if G_prev is None else np.stack([grad, G_prev])
        try:
            T, lam = qc.tangent(
                vectors, x, mask=free, return_multiplier=True, normals=N
            )
        except np.linalg.LinAlgError:
            return None
        if G_prev is None:
            G, PG_prev = T, None
        else:
            G, PG_prev, lam = T[0], T[1], lam[0]
        held = zero | (at_lo & (G >= 0.0)) | (at_hi & (G <= 0.0))
        Gp = np.where(held, 0.0, G)
        return held, lam, G, Gp, PG_prev

    def reproject(d):
        d = np.where(held, 0.0, d)
        try:
            return np.where(held, 0.0, qc.tangent(d, x, mask=~held, normals=N))
        except np.linalg.LinAlgError:
            return None

    def _finish(success, x, phi, grad, lam, nit, residual, message):
        violation = float(np.max(np.abs(qc.value(x) - qc.target)))
        return _result(
            success, np.asarray(s * x).reshape(original_shape), phi,
            np.asarray(grad / s).reshape(original_shape), lam, nit, residual,
            violation, message,
        )

    x = np.asarray(x0, dtype=float).ravel() / s
    restored = restore(x)
    if restored is None:
        raise RuntimeError(
            "l_bfgs_qc_projected: x0 cannot be restored onto the feasible set; "
            f"constraint residuals at x0: {qc.value(x) - qc.target}"
        )
    x, Bx = restored
    N = qc.normals(x, products=Bx)
    phi, grad = fun_grad(x)
    st = state(x, grad, N)
    if st is None:
        return _finish(
            False, x, phi, grad, np.full(qc.m, np.nan), 0, np.inf,
            "ABNORMAL: GRAM MATRIX OF THE CONSTRAINT NORMALS IS SINGULAR",
        )
    held, lam, G, Gp, _ = st
    r_norm = pnp.max(np.abs(Gp))

    s_hist, y_hist, rho_hist = [], [], []

    def _print(iteration, alpha):
        if disp and (comm is None or pnp is np or comm.rank == 0):
            lam_str = " ".join(f"{v:<12.4e}" for v in lam)
            print(f"{iteration:<5} {phi:<14.6e} {r_norm:<14.4e} {lam_str} {alpha:<8}")

    if disp and (comm is None or pnp is np or comm.rank == 0):
        print(f"{'iter':<5} {'f':<14} {'|Gp|inf':<14} {'lambda':<{13 * qc.m}} {'alpha':<8}")
    _print(0, "-")

    if r_norm < gtol:
        return _finish(
            True, x, phi, grad, lam, 0, r_norm,
            "CONVERGENCE: NORM_OF_PROJECTED_GRADIENT_<=_GTOL",
        )

    def steepest_descent():
        # Drop the curvature history and return the reprojected steepest-
        # descent direction with its slope, or None if the normals on the
        # entries that are not held are linearly dependent.
        s_hist.clear()
        y_hist.clear()
        rho_hist.clear()
        d = reproject(-Gp)
        return None if d is None else (d, pnp.sum(Gp * d))

    def line_search(d, gd, alpha):
        # Armijo backtracking along the restored arc. B (x + alpha d) is
        # B x + alpha B d, so one product of d serves every trial. A trial
        # that cannot be restored is rejected. The held entries stay on their
        # bounds: the restoration would otherwise pull them into the box by
        # O(alpha^2) and the set of held entries would never settle.
        Bd = qc.products(d)
        for _ in range(max_halvings):
            restored = restore(x + alpha * d, products=Bx + alpha * Bd, fixed=held)
            if restored is not None:
                x_try, Bx_try = restored
                phi_try, grad_try = fun_grad(x_try)
                if phi_try <= phi + c1 * alpha * gd:
                    return x_try, Bx_try, phi_try, grad_try, alpha
            alpha *= 0.5
        return None

    def _singular(iteration):
        return _finish(
            False, x, phi, grad, lam, iteration, r_norm,
            "ABNORMAL: GRAM MATRIX OF THE CONSTRAINT NORMALS IS SINGULAR",
        )

    for iteration in range(1, maxiter + 1):
        d = reproject(_twoloop_direction(Gp, s_hist, y_hist, rho_hist, pnp))
        gd = 0.0 if d is None else pnp.sum(Gp * d)
        if d is None or not gd < 0.0:
            # Stale curvature produced a non-descent direction: reset.
            fallback = steepest_descent()
            if fallback is None:
                return _singular(iteration)
            d, gd = fallback

        step = line_search(d, gd, 1.0)
        if step is None:
            fallback = steepest_descent()
            if fallback is None:
                return _singular(iteration)
            d, gd = fallback
            g_norm = np.sqrt(pnp.sum(Gp * Gp))
            step = line_search(d, gd, 1.0 / g_norm if g_norm > 0 else 1.0)
        if step is None:
            _log.info("line-search did not converge")
            return _finish(
                False, x, phi, grad, lam, iteration, r_norm,
                "CONVERGENCE: line-search did not converge",
            )
        x_new, Bx_new, phi_new, grad_new, alpha = step

        if iteration % _REFRESH_PRODUCTS == 0:
            Bx_new = qc.products(x_new)
        N_new = qc.normals(x_new, products=Bx_new)
        st_new = state(x_new, grad_new, N_new, G_prev=G)
        if st_new is None:
            return _singular(iteration)
        held_new, lam_new, G_new, Gp_new, PG = st_new

        # Curvature pair from the reduced gradient at x_new minus the previous
        # reduced gradient projected onto the tangent space at x_new.
        dx = x_new - x
        y = np.where(held_new, 0.0, G_new - PG)
        sy = pnp.sum(dx * y)
        dx_norm = np.sqrt(pnp.sum(dx * dx))
        y_norm = np.sqrt(pnp.sum(y * y))
        if sy > 1e-12 * (dx_norm * y_norm + np.finfo(float).eps):
            s_hist.insert(0, dx.copy())
            y_hist.insert(0, y.copy())
            rho_hist.insert(0, 1.0 / sy)
            if len(s_hist) > maxcor:
                s_hist.pop()
                y_hist.pop()
                rho_hist.pop()

        # Accept the step before testing for convergence, so that every exit
        # returns the iterate the line search accepted.
        dx_max = pnp.max(np.abs(dx))
        dphi = abs(phi_new - phi)
        phi_scale = max(1, abs(phi_new), abs(phi))
        x, Bx, N, phi, grad = x_new, Bx_new, N_new, phi_new, grad_new
        held, lam, G, Gp = held_new, lam_new, G_new, Gp_new
        r_norm = pnp.max(np.abs(Gp))

        _print(iteration, f"{alpha:.2e}")

        if callback is not None:
            callback((s * x).reshape(original_shape))

        if gtol > 0 and r_norm < gtol:
            return _finish(
                True, x, phi, grad, lam, iteration, r_norm,
                "CONVERGENCE: NORM_OF_PROJECTED_GRADIENT_<=_GTOL",
            )

        if xtol > 0 and dx_max < xtol:
            return _finish(
                True, x, phi, grad, lam, iteration, r_norm,
                "CONVERGENCE: NORM_OF_VARIABLE_STEP_<=_XTOL",
            )

        if ftol > 0 and dphi < ftol * phi_scale:
            return _finish(
                True, x, phi, grad, lam, iteration, r_norm,
                "CONVERGENCE: REL_REDUCTION_OF_F_<=_FTOL",
            )

    return _finish(
        False, x, phi, grad, lam, maxiter, r_norm,
        "NO CONVERGENCE: MAXITERATIONS REACHED",
    )


def _result(success, x, phi, grad, lam, nit, residual, violation, message):
    return OptimizeResult(
        {
            "success": bool(success),
            "x": x,
            "fun": phi,
            "jac": grad,
            "multiplier": np.asarray(lam, dtype=float),
            "nit": int(nit),
            "message": message,
            "max_grad": float(residual),
            "constr_violation": float(violation),
        }
    )
