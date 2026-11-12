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
from .QuadraticConstraint import BilinearConstraint, _active
from .Result import OptimizeResult

_log = logging.getLogger(__name__)

# The products B_j x are carried from one iterate to the next by linearity;
# they are recomputed after this many iterations to bound the rounding error.
_REFRESH_PRODUCTS = 20


def _balance_fields(fun_grad, x, grad, n, lo, hi, zero_mask, pnp, rng):
    r"""
    Scale that balances the curvature of the objective along the two fields
    of a bilinear constraint.

    The bilinear constraint does not change under :math:`g \to g / \sigma`,
    :math:`h \to \sigma h`. The curvature :math:`c` of the objective along
    each field is probed with one vector :math:`v` of entries :math:`\pm 1`
    on that field and a finite difference of the gradient,
    :math:`c = \langle v, \nabla f(x + \epsilon v) - \nabla f(x)\rangle
    / (\epsilon \langle v, v\rangle)`, the mean of the diagonal of the
    Hessian block of the field. In the variables :math:`g / \sigma` and
    :math:`\sigma h` the curvatures are :math:`c_g \sigma^2` and
    :math:`c_h / \sigma^2`, which are equal for
    :math:`\sigma = (c_h / c_g)^{1/4}`. Each probe stays inside the bounds.

    If a curvature is not positive, :math:`\sigma` instead equalises the
    norms of the two fields at ``x``, and it is 1 if a field vanishes.

    Args:
        fun_grad: ``fun_grad(x)`` returns the energy and the flat local
            gradient at the flat local ``x``.
        x: Flat local slice of a feasible point, :math:`g` followed by
            :math:`h`.
        grad: Gradient at ``x``.
        n: Local number of nodes.
        lo, hi: Flat bounds, scalar, array or ``None``.
        zero_mask: Flat boolean array of the entries pinned to zero.
        pnp: Reduction wrapper.
        rng: ``numpy.random.Generator`` of the probes.

    Returns:
        ``(scale, rule)``: the flat local scale, :math:`\sigma` on the
        entries of :math:`g` and :math:`1 / \sigma` on those of :math:`h`,
        and the rule that set it, ``"curvature"``, ``"norm"`` or ``"none"``.
    """
    lo_arr = np.broadcast_to(np.asarray(-np.inf if lo is None else lo, dtype=float), x.shape)
    hi_arr = np.broadcast_to(np.asarray(np.inf if hi is None else hi, dtype=float), x.shape)
    fields = (slice(0, n), slice(n, 2 * n))
    norm2 = [float(pnp.sum(x[field] ** 2)) for field in fields]
    count = float(pnp.sum(np.full(n, 1.0)))

    curvature = []
    for field, x2 in zip(fields, norm2):
        # Step of 1e-6 of the typical magnitude of the field, along random
        # signs that are flipped, or dropped, where the step would leave the
        # box.
        eps = 1e-6 * (np.sqrt(x2 / count) if x2 > 0.0 and count > 0.0 else 1.0)
        v = np.zeros(x.shape)
        v[field] = rng.choice([-1.0, 1.0], n)
        v = np.where(x + eps * v > hi_arr, -1.0, v)
        v = np.where((v < 0.0) & (x + eps * v < lo_arr), 1.0, v)
        v = np.where((x + eps * v > hi_arr) | (x + eps * v < lo_arr), 0.0, v)
        if zero_mask is not None:
            v = np.where(zero_mask, 0.0, v)
        v[fields[0] if field is fields[1] else fields[1]] = 0.0
        with np.errstate(all="ignore"):
            _, grad_probe = fun_grad(x + eps * v)
            vv = float(pnp.sum(v * v))
            c = float(pnp.sum(v * (grad_probe - grad))) / (eps * vv) if vv > 0.0 else np.nan
        curvature.append(c)

    c_g, c_h = curvature
    if np.isfinite(c_g) and np.isfinite(c_h) and c_g > 0.0 and c_h > 0.0:
        sigma, rule = (c_h / c_g) ** 0.25, "curvature"
    elif norm2[0] > 0.0 and norm2[1] > 0.0:
        sigma, rule = (norm2[0] / norm2[1]) ** 0.25, "norm"
    else:
        sigma, rule = 1.0, "none"
    if not (np.isfinite(sigma) and sigma > 0.0):
        sigma, rule = 1.0, "none"
    return np.concatenate([np.full(n, sigma), np.full(n, 1.0 / sigma)]), rule


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
            magnitude of each unknown. ``None`` means 1. ``"auto"``, for a
            ``BilinearConstraint`` only, sets ``s`` to ``sigma`` on the first
            field and ``1 / sigma`` on the second, which leaves the constraint
            unchanged, with ``sigma`` chosen such that the objective has the
            same mean curvature along both fields at the restored ``x0``. This
            takes two more evaluations of the gradient, at points inside the
            bounds. If a curvature is not positive, ``sigma`` equalises the
            norms of the two fields instead.
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
        ``max_grad``, ``constr_violation`` (the largest absolute residual
        of the constraint components at ``x``) and ``scale`` (the scale that
        was used, shaped like ``x``, or ``None``).

    Raises:
        RuntimeError: ``comm`` and ``pnp`` are both given, or ``x0`` cannot be
            restored onto the feasible set.
        ValueError: The constraint is missing, its reduction does not span
            the same ranks as the solver's, its ``hessp`` is not a symmetric
            linear map (``QuadraticConstraint.check_hessp``), ``scale`` is
            not positive and finite, or ``scale="auto"`` is asked for a
            constraint that is not a ``BilinearConstraint``.
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

    if jac is True or jac is None:

        def fun_grad_unscaled(x):
            phi, g = fun(x.reshape(original_shape), *args)
            return phi, np.asarray(g).ravel()

    elif jac is False:
        raise NotImplementedError("Numerical evaluation of gradient not implemented")
    else:

        def fun_grad_unscaled(x):
            xr = x.reshape(original_shape)
            return fun(xr, *args), np.asarray(jac(xr, *args)).ravel()

    # With a scale s the iteration runs on u = x / s: the objective gradient
    # becomes s * grad, the bounds lo / s and hi / s, and the constraint the
    # one in u. Below, `x` and `grad` are these scaled quantities.
    start = None
    if scale is None:
        s = 1.0
    elif isinstance(scale, str):
        if scale != "auto":
            raise ValueError(f"l_bfgs_qc_projected: unknown scale '{scale}'")
        if not isinstance(qc, BilinearConstraint):
            raise ValueError(
                'l_bfgs_qc_projected: scale="auto" balances the two fields of a '
                "BilinearConstraint; pass an array for other constraints"
            )
        # The curvatures are probed at the feasible starting point, which then
        # also starts the iteration: it stays feasible under this scale.
        x_start = np.asarray(x0, dtype=float).ravel()
        restored = qc.restore(
            x_start, lo=bounds_lo, hi=bounds_hi, zero_mask=zero_mask,
            max_rounds=max_restore_rounds,
        )
        if restored is None:
            raise RuntimeError(
                "l_bfgs_qc_projected: x0 cannot be restored onto the feasible set; "
                f"constraint residuals at x0: {qc.value(x_start) - qc.target}"
            )
        x_start = restored[0]
        phi_start, grad_start = fun_grad_unscaled(x_start)
        rank = 0 if comm is None else comm.Get_rank()
        s, rule = _balance_fields(
            fun_grad_unscaled, x_start, grad_start, qc.n, bounds_lo, bounds_hi,
            zero_mask, pnp, np.random.default_rng([rank, 20261112]),
        )
        _log.info(f"scale='auto': sigma = {s[0] if s.size else 1.0:g} from rule '{rule}'")
        start = (x_start / s, phi_start, s * grad_start)
    else:
        s = np.broadcast_to(
            np.asarray(scale, dtype=float), original_shape
        ).ravel().copy()
        if not (np.all(np.isfinite(s)) and np.all(s > 0.0)):
            raise ValueError("l_bfgs_qc_projected: scale must be positive and finite")
    if scale is not None:
        qc = qc.scaled(s)
        bounds_lo = None if bounds_lo is None else bounds_lo / s
        bounds_hi = None if bounds_hi is None else bounds_hi / s

    def fun_grad(x):
        phi, g = fun_grad_unscaled(s * x)
        return phi, s * g

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
            None if scale is None else np.broadcast_to(s, x.shape).reshape(original_shape),
        )

    if start is None:
        x = np.asarray(x0, dtype=float).ravel() / s
        restored = restore(x)
        if restored is None:
            raise RuntimeError(
                "l_bfgs_qc_projected: x0 cannot be restored onto the feasible set; "
                f"constraint residuals at x0: {qc.value(x) - qc.target}"
            )
        x, Bx = restored
        phi, grad = fun_grad(x)
    else:
        x, phi, grad = start
        Bx = qc.products(x)
    N = qc.normals(x, products=Bx)
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


def _result(success, x, phi, grad, lam, nit, residual, violation, message, scale):
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
            "scale": scale,
        }
    )
