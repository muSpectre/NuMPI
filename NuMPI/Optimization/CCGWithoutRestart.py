"""
Constrained Conjugate Gradient for moderately nonlinear problems.

This module implements the bound constrained conjugate gradient algorithm
from Bugnicourt et al. (2018), designed for solving contact mechanics problems
with inequality constraints. The algorithm is MPI-parallelized and supports
mean value constraints.

This implementation is mainly based upon Bugnicourt et. al. - 2018, OUT_LIN
algorithm.
"""

from inspect import signature

import numpy as np

from ..Tools import Reduction
from .Result import OptimizeResult


def constrained_conjugate_gradients(fun, hessp, x0, args=(), mean_val=None, gtol=1e-8, maxiter=3000, callback=None,
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
        The objective function to be minimized. Must have signature
        ``fun(x, *args) -> (energy, gradient)`` where energy is a float
        and gradient is an ndarray of the same shape as x. Note that the
        energy value is not used by this algorithm; you can return a dummy
        value (e.g., 0.0).
    hessp : callable
        Function to evaluate the Hessian-vector product. Must accept either:
        - 1 argument: ``hessp(descent_direction) -> ndarray``
        - 2 arguments: ``hessp(x, descent_direction) -> ndarray``
        The function should return the product of the Hessian matrix with
        the descent direction vector.
    x0 : ndarray
        Initial guess for the solution. Must not be None.
    args : tuple, optional
        Extra arguments passed to `fun`. Default is ().
    mean_val : float, optional
        If provided, enforces a mean value constraint on the solution.
        The algorithm will adjust the solution to maintain this mean value
        over the non-bounded (active) region. Only compatible with fully
        bounded systems (all elements must have finite bounds).
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

    Notes
    -----
    The algorithm maintains complementarity conditions: for each element i,
    either x[i] > bounds[i] (inactive constraint, zero Lagrange multiplier)
    or the projected gradient is non-negative (active constraint).

    When `mean_val` is specified, the algorithm enforces that the mean of
    the solution over the non-bounded region equals `mean_val`. This is
    useful for problems with integral constraints.

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
    mean_bounds = comm.sum(bounds) / nb_bounds

    if mean_val is not None and nb_bounds < nb_DOF:
        raise ValueError("mean_value constrained mode not compatible "
                         "with partially bound system")
        # There are ambiguities on how to compute the mean values

    '''Initial Residual = A^(-1).(U) - d A −1 .(U ) −  ∂ψadh/∂g'''
    residual = fun(x, *args)[1]

    mask_neg = x <= bounds
    x[mask_neg] = bounds[mask_neg]

    if mean_val is not None:
        #
        mask_nonzero = x > bounds
        N_mask_nonzero = comm.sum(np.count_nonzero(mask_nonzero))
        residual = residual - comm.sum(residual[mask_nonzero]) / N_mask_nonzero

    '''Apply the admissible Lagrange multipliers.'''
    mask_res = residual >= 0
    mask_bounded = np.logical_and(mask_neg, mask_res)
    residual[mask_bounded] = 0.0

    if mean_val is not None:
        #
        mask_nonzero = residual != 0
        N_nonzero = comm.sum(np.count_nonzero(mask_nonzero))
        residual[mask_nonzero] = residual[mask_nonzero] - comm.sum(
            residual[mask_nonzero]) / N_nonzero
    '''INITIAL DESCENT DIRECTION'''
    des_dir = -residual

    n_iterations = 1

    for i in range(1, maxiter + 1):
        sig = signature(hessp)
        if len(sig.parameters) == 2:
            hessp_val = hessp(x, des_dir)
        elif len(sig.parameters) == 1:
            hessp_val = hessp(des_dir)
        else:
            raise ValueError('hessp function has to take max 1 arg (descent '
                             'dir) or 2 args (x, descent direction)')
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

        if mean_val is not None:
            # x = (mean_val / comm.sum(x) * nb_DOF) * x
            # below is just a more complicated version of this compatible with
            # more general bounds
            x = bounds + (mean_val - mean_bounds) \
                / (comm.sum(x) / nb_DOF - mean_bounds) * (x - bounds)
        residual_old = residual

        '''
        In Bugnicourt's paper
        Residual = A^(-1).(U) - d A −1 .(U ) −  ∂ψadh/∂g
        '''
        residual = fun(x, *args)[1]

        if mean_val is not None:
            mask_nonzero = x > bounds
            N_mask_nonzero = comm.sum(np.count_nonzero(mask_nonzero))
            residual = residual - comm.sum(
                residual[mask_nonzero]) / N_mask_nonzero

        '''Apply the admissible Lagrange multipliers.'''
        mask_res = residual >= 0
        mask_bounded = np.logical_and(mask_neg, mask_res)
        residual[mask_bounded] = 0.0

        if mean_val is not None:
            mask_nonzero = residual != 0
            N_nonzero = comm.sum(np.count_nonzero(mask_nonzero))
            residual[mask_nonzero] = residual[mask_nonzero] - comm.sum(
                residual[mask_nonzero]) / N_nonzero
            # assert np.mean(residual) < 1e-14 * np.max(abs(residual))

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
