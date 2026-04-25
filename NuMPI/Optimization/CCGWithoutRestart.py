"""
Constrained Conjugate Gradient for moderately nonlinear problems.

This module implements the bound constrained conjugate gradient algorithm
from Bugnicourt et al. (2018), designed for solving contact mechanics problems
with inequality constraints. The algorithm is MPI-parallelized and supports
an optional linear equality constraint (e.g. a prescribed mean value, or more
generally a weighted integral constraint such as a fixed liquid volume in a
phase-field model).

This implementation is mainly based upon Bugnicourt et. al. - 2018, OUT_LIN
algorithm.
"""

from inspect import signature

import numpy as np

from ..Tools import Reduction
from .LinearConstraint import LinearConstraint
from .Result import OptimizeResult


def constrained_conjugate_gradients(fun, hessp, x0, args=(), jac=True,
                                    mean_val=None, linear_constraint=None,
                                    gtol=1e-8, maxiter=3000, callback=None,
                                    communicator=None, bounds=None):
    """
    Constrained conjugate gradient algorithm from Bugnicourt et al. [1].

    This algorithm solves bound-constrained quadratic optimization problems
    using a conjugate gradient method with active set constraints. It is
    particularly suited for contact mechanics problems where the solution
    must satisfy inequality constraints (e.g., non-penetration).

    The algorithm iteratively updates the solution while projecting onto
    the feasible set defined by the bounds. It uses a Polak-Ribière-like
    formula for updating the conjugate direction.

    Parameters
    ----------
    fun : callable
        Objective function input. Interpretation depends on ``jac``:

        - ``jac=True`` (default): ``fun(x, *args) -> (energy, gradient)``
        - ``jac=False``: ``fun(x, *args) -> gradient``
        - ``jac`` callable: ``fun(x, *args) -> energy`` and
          ``jac(x, *args) -> gradient``

        The energy value is not used by this algorithm; in ``jac=True`` mode,
        returning a dummy scalar (e.g., ``0.0``) is fine.
    hessp : callable
        Function to evaluate the Hessian-vector product. Accepted signatures
        are:
        - ``hessp(descent_direction) -> ndarray``
        - ``hessp(x, descent_direction) -> ndarray``
        - ``hessp(descent_direction, *args) -> ndarray``
        - ``hessp(x, descent_direction, *args) -> ndarray``
        where ``*args`` are the optional extra arguments passed via ``args``.
        The function should return the product of the Hessian matrix with the
        descent direction vector.
    x0 : ndarray
        Initial guess for the solution. Must not be None.
    args : tuple, optional
        Extra arguments passed to `fun`. Default is ().
    jac : bool or callable, optional
        Controls how gradients are obtained from ``fun``:

        - ``True``: treat ``fun`` as SciPy-style ``(f, g)`` function.
        - ``False``: treat ``fun`` as a gradient-only function.
        - callable: use separate gradient callback ``jac(x, *args)``.
    mean_val : float, optional
        If provided, enforces a mean value constraint on the solution over
        the non-bounded (active) region. Equivalent to passing
        ``linear_constraint=LinearConstraint(np.ones_like(x0), mean_val * N)``
        where ``N`` is the global number of degrees of freedom. Only
        compatible with fully bounded systems (all elements must have finite
        bounds). Provided for backwards compatibility; prefer
        ``linear_constraint`` for new code.
    linear_constraint : LinearConstraint, optional
        A general linear equality constraint ``<a, x> = target``. When set,
        the descent direction is tangent-projected against ``a`` after each
        gradient evaluation and the iterate is projected back onto
        ``{x : <a, x> = target, x >= bounds}`` after each line step. Mutually
        exclusive with ``mean_val``.
    gtol : float, optional
        Gradient tolerance for convergence. The algorithm converges when
        ``max(abs(projected_gradient)) <= gtol``. Default is 1e-8.
    maxiter : int, optional
        Maximum number of iterations. Default is 3000.
    callback : callable, optional
        Function called after each iteration with the current solution x.
        Signature: ``callback(x) -> None``.
    communicator : MPI.Comm, optional
        MPI communicator for parallel execution. If None, runs in serial
        mode using numpy operations.
    bounds : ndarray, optional
        Lower bounds for each element of x. Elements of x will be constrained
        to be >= bounds. If None, defaults to zero bounds (equivalent to
        non-negativity constraints).

    Returns
    -------
    result : OptimizeResult
        The optimization result with the following attributes:

        - success : bool
            True if the algorithm converged within the tolerance.
        - x : ndarray
            The optimized solution.
        - jac : ndarray
            The residual (projected gradient) at the solution.
        - nit : int
            Number of iterations performed.
        - message : str
            Description of the termination reason.

    Raises
    ------
    ValueError
        If `hessp` has an unsupported number of parameters (must be 1 or 2).
        If `mean_val` is provided but the system is only partially bounded.
        If both `mean_val` and `linear_constraint` are set.

    Notes
    -----
    The algorithm maintains complementarity conditions: for each element i,
    either x[i] > bounds[i] (inactive constraint, zero Lagrange multiplier)
    or the projected gradient is non-negative (active constraint).

    When a linear equality constraint is supplied (either ``mean_val`` or
    ``linear_constraint``), the algorithm additionally enforces
    ``<a, x> = target`` by: (i) removing the tangential component
    ``lambda * a`` from the residual (with ``lambda`` restricted to the
    non-bounded indices), and (ii) projecting ``x`` back onto the constraint
    hyperplane after each line step.

    **MPI contract.** ``x0``, ``bounds``, and ``linear_constraint.a`` are
    *local* per-rank slices; the gradient returned by ``fun`` is local; the
    scalar energy returned by ``fun`` (if any) and ``linear_constraint.target``
    are **global**. See the "MPI Conventions" section of the top-level README
    for details and a worked example.

    References
    ----------
    .. [1] Bugnicourt, Romain & Sainsot, Philippe & Dureisseix, David &
        Gauthier, Catherine & Lubrecht, Ton. (2018). FFT-Based Methods
        for Solving a Rough Adhesive Contact: Description and
        Convergence Study. Tribology International, 121, 200-209.
    """
    if communicator is None:
        comm = np
        nb_DOF = x0.size
    else:
        comm = Reduction(communicator)
        nb_DOF = comm.sum(x0.size)

    x = x0.copy()
    x = x.flatten()

    if bounds is None:
        bounds = np.zeros_like(x)

    mask_bounds = bounds > - np.inf
    nb_bounds = comm.sum(np.count_nonzero(mask_bounds))

    # --- Normalise the linear-equality constraint --------------------------
    # Accept either the old `mean_val` convenience kwarg (uniform weights) or
    # a generic LinearConstraint. When set, run the projection / tangent
    # steps in the iteration loop below.
    if mean_val is not None and linear_constraint is not None:
        raise ValueError(
            "Pass either mean_val or linear_constraint, not both."
        )
    if mean_val is not None and nb_bounds < nb_DOF:
        raise ValueError("mean_value constrained mode not compatible "
                         "with partially bound system")
        # There are ambiguities on how to compute the mean values
    if mean_val is not None:
        linear_constraint = LinearConstraint(np.ones_like(x),
                                             target=float(mean_val) * nb_DOF,
                                             pnp=comm)
    constrained = linear_constraint is not None

    if jac is True:
        grad = lambda x_: fun(x_, *args)[1]  # noqa: E731
    elif jac is False:
        grad = lambda x_: fun(x_, *args)  # noqa: E731
    elif callable(jac):
        grad = lambda x_: jac(x_, *args)  # noqa: E731
    else:
        raise ValueError("jac must be True, False, or a callable")

    '''Initial Residual = A^(-1).(U) - d A −1 .(U ) −  ∂ψadh/∂g'''
    residual = grad(x)

    mask_neg = x <= bounds
    x[mask_neg] = bounds[mask_neg]

    if constrained:
        # Tangent-project the residual over the free (non-bounded) indices:
        # res <- res - lambda * a  with lambda = <a_free, res_free>/<a_free, a_free>.
        mask_free = x > bounds
        residual = linear_constraint.tangent(residual, mask=mask_free)

    '''Apply the admissible Lagrange multipliers.'''
    mask_res = residual >= 0
    mask_bounded = np.logical_and(mask_neg, mask_res)
    residual[mask_bounded] = 0.0

    if constrained:
        # Zeroing the residual at bound-active indices breaks tangent
        # orthogonality; re-project over indices that survived.
        mask_active = residual != 0
        residual = linear_constraint.tangent(residual, mask=mask_active)

    '''INITIAL DESCENT DIRECTION'''
    des_dir = -residual

    n_iterations = 1

    hessp_nargs = len(signature(hessp).parameters)

    for i in range(1, maxiter + 1):
        if hessp_nargs == 2 + len(args):
            hessp_val = hessp(x, des_dir, *args)
        elif hessp_nargs == 1 + len(args):
            hessp_val = hessp(des_dir, *args)
        elif hessp_nargs == 2:
            hessp_val = hessp(x, des_dir)
        elif hessp_nargs == 1:
            hessp_val = hessp(des_dir)
        else:
            raise ValueError(
                'Unsupported hessp signature. Expected one of '
                '(d), (x, d), (d, *args), or (x, d, *args).'
            )
        denominator_temp = comm.sum(des_dir.T * hessp_val)
        # Here we could evaluate this directly in Fourier space (Parseval)
        # and spare one FFT.
        # See issue #47

        if denominator_temp == 0:
            print("it {}: denominator for alpha is 0".format(i))

        alpha = -comm.sum(residual.T * des_dir) / denominator_temp

        if alpha < 0:
            print("it {} : hessian is negative along the descent direction. "
                  "You will probably need linesearch "
                  "or trust region".format(i))

        x += alpha * des_dir

        '''finding new contact points and making the new_gap admissible
        according to these contact points.'''

        mask_neg = x <= bounds
        x[mask_neg] = bounds[mask_neg]

        if constrained:
            # Euclidean projection onto {x : <a, x> = target, x >= bounds}.
            # Upper bound is unused (hi=None is treated as +infinity).
            x = linear_constraint.project(x, lo=bounds)
            # Refresh the bound-active mask — the projection may pull nodes
            # back above `bounds` or push others down to them.
            mask_neg = x <= bounds
        residual_old = residual

        '''
        In Bugnicourt's paper
        Residual = A^(-1).(U) - d A −1 .(U ) −  ∂ψadh/∂g
        '''
        residual = grad(x)

        if constrained:
            mask_free = x > bounds
            residual = linear_constraint.tangent(residual, mask=mask_free)

        '''Apply the admissible Lagrange multipliers.'''
        mask_res = residual >= 0
        mask_bounded = np.logical_and(mask_neg, mask_res)
        residual[mask_bounded] = 0.0

        if constrained:
            mask_active = residual != 0
            residual = linear_constraint.tangent(residual, mask=mask_active)

        '''Computing beta for updating descent direction
            In Bugnicourt's paper:
            beta = num / denom
            num = new_residual_transpose . (new_residual - old_residual)
            denom = alpha * descent_dir_transpose . (A_inverse - d2_ψadh).
            descent_dir '''

        # beta = np.sum(residual.T * hessp_val) / denominator_temp
        beta = comm.sum(residual * (residual - residual_old)) / (
                alpha * denominator_temp)

        des_dir_old = des_dir
        des_dir = -residual + beta * des_dir_old

        des_dir[mask_bounded] = 0

        if callback:
            callback(x)

        n_iterations += 1

        if comm.max(abs(residual)) <= gtol:
            result = OptimizeResult(
                {
                    'success': True,
                    'x': x,
                    'jac': residual,
                    'nit': i,
                    'message': 'CONVERGENCE: NORM_OF_GRADIENT_<=_GTOL',
                })
            return result

        elif i == maxiter:
            result = OptimizeResult(
                {
                    'success': False,
                    'x': x,
                    'jac': residual,
                    'nit': i,
                    'message': 'NO CONVERGENCE: MAXITERATIONS REACHED'
                })

            return result
