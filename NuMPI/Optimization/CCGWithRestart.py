"""
Constrained Conjugate Gradient with restart on active-set changes.

This module implements the bound-constrained conjugate gradient algorithm of
Polonsky & Keer (1999) for contact-mechanics-type problems with a
non-penetration (lower-bound) inequality constraint and an optional mean-value
equality constraint.

Unlike the Bugnicourt et al. variant in :mod:`CCGWithoutRestart`, the conjugate
direction is **restarted** (reset to steepest descent) whenever the active
(contact) set changes -- this is the defining feature of the Polonsky-Keer
scheme and is controlled by the ``delta`` flag below. The algorithm is
MPI-parallelized: distributed arrays are sliced per rank and every scalar that
steers the iteration (step length, restart decision, convergence test) is
globally reduced so all ranks remain in lock-step.

References
----------
.. [1] I.A. Polonsky, L.M. Keer, "A numerical method for solving rough contact
    problems based on the multi-level multi-summation and conjugate gradient
    techniques", Wear 231, 206 (1999).
"""

from inspect import signature

import numpy as np

from ..Tools import Reduction
from .Result import OptimizeResult


def constrained_conjugate_gradients_with_restart(
        fun, hessp, x0, args=(), jac=True, mean_val=None, gtol=1e-8,
        maxiter=5000, callback=None, communicator=None, bounds=None,
        residual_plot=False):
    """
    Constrained conjugate gradient with restart (Polonsky & Keer [1]).

    Solves a bound-constrained quadratic problem ``min f(x)`` subject to
    ``x >= bounds`` (and optionally a prescribed mean value of ``x``). The
    conjugate direction is reset to steepest descent whenever the active set
    changes between iterations.

    Parameters
    ----------
    fun : callable
        Objective. Interpretation depends on ``jac``:

        - ``jac=True`` (default): ``fun(x, *args) -> (energy, gradient)``.
          The energy is unused; a dummy scalar is fine.
        - ``jac=False``: ``fun(x, *args) -> gradient``.
        - ``jac`` callable: ``fun`` is ignored for gradients and
          ``jac(x, *args) -> gradient`` is used.
    hessp : callable
        Hessian-vector product. Accepted signatures are ``hessp(d)``,
        ``hessp(x, d)``, ``hessp(d, *args)`` or ``hessp(x, d, *args)``, where
        ``d`` is the descent direction and ``*args`` are the extra arguments
        passed via ``args``.
    x0 : ndarray
        Initial guess (the local per-rank slice in MPI mode). Must not be
        ``None``.
    args : tuple, optional
        Extra arguments forwarded to ``fun`` / ``jac`` / ``hessp``.
    jac : bool or callable, optional
        See ``fun``. Default ``True``.
    mean_val : float, optional
        If given, enforce a prescribed global mean value of ``x``. The search
        direction is projected onto the constraint tangent (its mean over the
        free indices is removed) and ``x`` is renormalised after each step.
    gtol : float, optional
        Convergence tolerance on the infinity norm of the residual over the
        free (non-bound-active) indices. Default ``1e-8``.
    maxiter : int, optional
        Maximum number of iterations. Default ``5000``.
    callback : callable, optional
        Called after each iteration as ``callback(x)`` with the current
        (local) iterate.
    communicator : MPI.Comm, optional
        MPI communicator. If ``None``, runs serially using plain numpy
        reductions.
    bounds : ndarray, optional
        Lower bounds (local slice). ``x`` is constrained to ``x >= bounds``.
        ``None`` defaults to zeros (non-negativity). Use ``-inf`` entries for
        unbounded components.
    residual_plot : bool, optional
        If ``True``, plot the residual history on rank 0 at convergence
        (requires matplotlib). Default ``False``.

    Returns
    -------
    result : OptimizeResult
        With attributes ``success`` (bool), ``x`` (solution), ``jac``
        (residual), ``nit`` (iteration count) and ``message``.

    Raises
    ------
    ValueError
        If ``x0`` is ``None``, ``mean_val`` has the wrong type, or ``hessp``
        has an unsupported signature.

    Notes
    -----
    **MPI contract.** ``x0`` and ``bounds`` are *local* per-rank slices and the
    gradient returned by ``fun`` is local; the scalar ``mean_val`` and ``gtol``
    are *global* (identical on every rank). All inner products, the restart
    decision and the convergence test are globally reduced. See the "MPI
    Conventions" section of the top-level README.

    References
    ----------
    .. [1] I.A. Polonsky, L.M. Keer, Wear 231, 206 (1999).
    """
    if not isinstance(mean_val, (type(None), int, float)):
        raise ValueError(
            'Inappropriate type: {} for mean_val whereas a float or int is '
            'expected'.format(type(mean_val)))
    if not isinstance(residual_plot, bool):
        raise ValueError(
            'Inappropriate type: {} for "residual_plot" whereas a bool is '
            'expected'.format(type(residual_plot)))
    if x0 is None:
        raise ValueError('Input required for x0/initial value !!')

    if communicator is None:
        comm = np
    else:
        comm = Reduction(communicator)

    x = x0.copy().flatten()
    nb_DOF = comm.sum(x.size)

    if bounds is None:
        bounds = np.zeros_like(x)

    if jac is True:
        grad = lambda x_: fun(x_, *args)[1]  # noqa: E731
    elif jac is False:
        grad = lambda x_: fun(x_, *args)  # noqa: E731
    elif callable(jac):
        grad = lambda x_: jac(x_, *args)  # noqa: E731
    else:
        raise ValueError("jac must be True, False, or a callable")

    hessp_nargs = len(signature(hessp).parameters)

    def apply_hessp(d):
        if hessp_nargs == 2 + len(args):
            return hessp(x, d, *args)
        elif hessp_nargs == 1 + len(args):
            return hessp(d, *args)
        elif hessp_nargs == 2:
            return hessp(x, d)
        elif hessp_nargs == 1:
            return hessp(d)
        raise ValueError(
            'Unsupported hessp signature. Expected one of (d), (x, d), '
            '(d, *args), or (x, d, *args).')

    def free_mean(v, mask, nb_free):
        """Global mean of ``v`` over the free indices ``mask``."""
        if nb_free == 0:
            return 0.0
        return comm.sum(v[mask]) / nb_free

    delta = 0
    G_old = 1.0
    des_dir = np.zeros_like(x)

    gaps = []
    iterations = []

    for i in range(1, maxiter + 1):
        # Truncate to the feasible set and identify the free (open) indices.
        mask_neg = x <= bounds
        x[mask_neg] = bounds[mask_neg]

        residual = grad(x)

        mask_c = x > bounds
        nb_free = comm.sum(np.count_nonzero(mask_c))

        if mean_val is not None:
            # Project the residual onto the constraint tangent (remove its
            # global mean over the free indices).
            residual = residual - free_mean(residual, mask_c, nb_free)

        # Conjugate-direction update. delta == 0 restarts (steepest descent);
        # delta == 1 keeps conjugacy. G is a globally reduced inner product.
        # Guard G_old == 0 (empty free set last step) so a restart stays a
        # genuine steepest-descent step rather than 0 * inf = nan.
        G = comm.sum(residual[mask_c] ** 2)
        beta = (G / G_old) if (delta == 1 and G_old != 0) else 0.0
        des_dir[mask_c] = -residual[mask_c] + beta * des_dir[mask_c]
        des_dir[np.logical_not(mask_c)] = 0
        G_old = G

        # Step length alpha (TAU in the paper) from globally reduced products.
        hessp_val = apply_hessp(des_dir)
        if mean_val is not None:
            hessp_val = hessp_val - free_mean(hessp_val, mask_c, nb_free)

        if nb_free != 0:
            denominator = comm.sum(hessp_val[mask_c] * des_dir[mask_c])
            if denominator == 0:
                if communicator is None or communicator.rank == 0:
                    print("it {}: denominator for alpha is 0".format(i))
                alpha = 0.0
            else:
                alpha = -comm.sum(
                    residual[mask_c] * des_dir[mask_c]) / denominator
        else:
            alpha = 0.0

        if alpha < 0 and (communicator is None or communicator.rank == 0):
            print("it {} : hessian is negative along the descent direction. "
                  "You will probably need linesearch or trust region"
                  .format(i))

        x[mask_c] += alpha * des_dir[mask_c]

        # New contact points; relax the iterate there and decide whether to
        # restart. The restart decision MUST be globally consistent.
        mask_neg = x <= bounds
        x[mask_neg] = bounds[mask_neg]

        mask_g = residual < 0
        mask_overlap = np.logical_and(mask_neg, mask_g)

        if comm.sum(np.count_nonzero(mask_overlap)) == 0:
            delta = 1
        else:
            delta = 0
            x[mask_overlap] = x[mask_overlap] - alpha * residual[mask_overlap]

        if mean_val is not None:
            # Enforce the prescribed global mean value of x.
            mean_x = comm.sum(x) / nb_DOF
            if mean_x != 0:
                x *= mean_val / mean_x

        assert not np.isnan(x).any()

        if callback is not None:
            callback(x)

        # KKT/projected residual: free indices must have ~zero gradient; a
        # bound-active index is optimal when its gradient is non-negative (the
        # bound is pushing back). Checking this over the whole domain -- rather
        # than only the free set -- lets the solver converge even when the
        # solution is fully bound-active.
        proj_res = np.where(mask_c, np.abs(residual),
                            np.maximum(-residual, 0.0))
        max_residual = comm.max(proj_res)

        if residual_plot:
            iterations.append(i)
            gaps.append(max_residual)

        if max_residual <= gtol:
            if residual_plot:
                _plot_residuals(iterations, gaps, communicator)
            return OptimizeResult({
                'success': True,
                'x': x,
                'jac': residual,
                'nit': i,
                'message': 'CONVERGENCE: NORM_OF_GRADIENT_<=_GTOL',
            })

        if i == maxiter:
            if residual_plot:
                _plot_residuals(iterations, gaps, communicator)
            return OptimizeResult({
                'success': False,
                'x': x,
                'jac': residual,
                'nit': i,
                'message': 'NO CONVERGENCE: MAXITERATIONS REACHED',
            })


def _plot_residuals(iterations, gaps, communicator):
    if communicator is not None and communicator.rank != 0:
        return
    import matplotlib.pyplot as plt
    plt.plot(iterations, np.log10(gaps))
    plt.xlabel('iterations')
    plt.ylabel('residuals')
    plt.show()
