#
# Copyright 2018, 2020 Antoine Sanner
#           2019 Lars Pastewka
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

# skip undefined names
# flake8: noqa F821


import numpy as np

from .. import MPI
from ..Tools import Reduction
from .LineSearch import scalar_search_wolfe2
from .Result import OptimizeResult


def donothing(*args, **kwargs):
    pass


def steepest_descent_wolfe2(x0, f_gradf, pnp=None, maxiter=10, args=(), **kwargs):
    """
    For first Iteration there is no history. We make a steepest descent
    satisfying strong Wolfe Condition
    :return:
    """
    # x_old.shape=(-1,1)
    phi0, grad0 = f_gradf(x0, *args)
    grad0 = grad0.reshape((-1, 1))

    derphi0 = pnp.dot(grad0.T, -grad0).item()
    gradf = np.zeros_like(grad0)

    # stores the last evaluated gradf,
    # take care, the linesearch has to return the last evaluated value
    # TODO: Dangerous
    def _phi_phiprime(alpha):
        phi, gradf[...] = f_gradf(x0 - grad0 * alpha, *args)
        phiprime = pnp.dot(gradf.T, -grad0).item()
        return phi, phiprime

    # like in lbfgsb
    # if (iter.eq. 0.and..not.boxed) then
    #    stp = min(one / dnorm, stpmx)

    alpha, phi, phi0, derphi = scalar_search_wolfe2(_phi_phiprime,
                                                    maxiter=maxiter,
                                                    phi0=phi0, derphi0=derphi0,
                                                    step=min(1., 1. / np.sqrt(
                                                        -derphi0)), **kwargs)

    assert derphi is not None, "Line Search in first steepest descent failed"
    x = x0 - grad0 * alpha

    return x, gradf, x0, grad0, phi, phi0, derphi


def l_bfgs(fun, x, args=(), jac=None, x_old=None, maxcor=10, gtol=1e-5, ftol=2.2e-9, maxiter=15000, maxls=20,
           linesearch_options=dict(c1=1e-3, c2=0.9), disp=None, pnp=Reduction(MPI.COMM_WORLD), store_iterates=None,
           printdb=donothing, callback=None, **options):
    """
    Limited-memory L-BFGS optimization with MPI parallelization.

    The optimization converges if |grad|_{\infty} <= gtol or <= ftol is satisfied.

    Parameters
    ----------
    fun : callable
        The objective function to be minimized.
    x : ndarray
        Initial guess.
    args : tuple, optional
        Extra arguments passed to the objective function and its derivatives (Jacobian).
    jac : bool or callable, optional
        Method for computing the gradient vector. If it is a callable, it should be a function that returns the gradient vector. If it is a Boolean and is True, then `fun` should return the gradient along with the objective function.
    x_old : ndarray, optional
        Initial guess for the previous value of x. If None, a steepest descent step will be performed.
    maxcor : int, optional
        The maximum number of variable metric corrections used to define the limited memory matrix.
    gtol : float, optional
        The iteration will stop when max{|proj g_i | i = 1, ..., n} <= gtol where pg_i is the i-th component of the projected gradient.
    ftol : float, optional
        The iteration will stop when (f^k - f^{k+1})/max{|f^k|,|f^{k+1}|,1} <= ftol.
    maxiter : int, optional
        Maximum number of iterations.
    maxls : int, optional
        Maximum number of line search steps (per iteration).
    linesearch_options : dict, optional
        A dictionary with optional settings for the line search parameters.
    disp : int, optional
        If disp is a positive integer, convergence messages will be printed.
    pnp : Reduction, optional
        Parallelization object from the NuMPI package.
    store_iterates : bool, optional
        If True, the result at each iteration is stored in a list.
    printdb : callable, optional
        Debugging print function.
    callback : callable, optional
        Called after each iteration.
    options : dict, optional
        A dictionary with optional settings for the optimization algorithm.

    Returns
    -------
    result : OptimizeResult
        The optimization result represented as a OptimizeResult object. Important attributes are: x the solution array, success a Boolean flag indicating if the optimizer exited successfully and message which describes the cause of the termination.
    """
    # user can provide x in the shape of his convenience
    # but we will work with the shape (-1, 1)
    original_shape = x.shape

    if callback is None:
        callback = donothing

    # print("jac = {}, type = {}".format(jac, type(jac)))
    if jac is True:
        def fun_grad(x, *args):
            """
            This function evaluates the objective function and its gradient at the point x.
            It reshapes the gradient into a convenient form for further computations.

            Parameters
            ----------
            x : ndarray
                The point at which the objective function and its gradient are to be evaluated.

            Returns
            -------
            f : float
                The value of the objective function at the point x.
            grad : ndarray
                The gradient of the objective function at the point x, reshaped into a convenient form.
            """
            f, grad = fun(x.reshape(original_shape), *args)
            return f, grad.reshape((-1, 1))

    elif jac is False:
        raise NotImplementedError(
            "Numerical evaluation of gradient not implemented")
    else:
        # function and gradient provided sepatately

        def fun_grad(x, *args):
            """
            This function evaluates the objective function and its gradient at the point x.
            It reshapes the gradient into a convenient form for further computations.
            The order of function and gradient evaluation is important (see issue #13).

            Parameters
            ----------
            x : ndarray
                The point at which the objective function and its gradient are to be evaluated.

            Returns
            -------
            f : float
                The value of the objective function at the point x.
            grad : ndarray
                The gradient of the objective function at the point x, reshaped into a convenient form.
            """
            # order matte
            f = fun(x.reshape(original_shape), *args)
            grad = jac(x.reshape(original_shape), *args).reshape((-1, 1))
            return f, grad

    x = x.reshape((-1, 1))

    if x_old is None:
        x_old = x.copy()
        x, grad, x_old, grad_old, phi, phi_old, derphi = \
            steepest_descent_wolfe2(x_old, fun_grad, pnp=pnp, maxiter=maxls,
                                    args=args, **linesearch_options)
    else:
        # phi_old is never used, except for the convergence
        phi_old, grad_old = fun_grad(x_old, *args)
        # criterion
        phi, grad = fun_grad(x, *args)

    # full history of x is sored here if wished
    iterates = list()
    iteration = 1

    # n = x.size  # number of degrees of freedom
    gamma = 1

    Slist = []  # history of the steps of x
    Ylist = []  # history of gradient differences
    R = np.zeros((0, 0))

    grad2 = pnp.sum(grad ** 2)

    alpha = 0  # line search step size

    if disp:
        if MPI.COMM_WORLD.rank == 0:
            iteration_title = "iteration"
            phi_title = "f"
            phi_change_title = "Δf"
            max_grad_title = "max ∇ f"
            abs_grad_title = "|∇ f|"
            print(
                f"{iteration_title:<10} {phi_title:<10} {phi_change_title:<10} {max_grad_title:<10} {abs_grad_title:<10}")
            print(f"---------- ---------- ---------- ---------- ----------")

    # Start loop
    # printdb(k)
    while True:
        # Update Sk,Yk
        # print("k= {}".format(k))
        if iteration > maxcor:
            # S = np.roll(S, -1)
            # S[:, -1] = (x - x_old).flat

            Slist[:-1] = Slist[1:]
            Slist[-1] = (x - x_old)

            # Y = np.roll(Y, -1)
            # Y[:, -1] = (grad - grad_old).flat

            Ylist[:-1] = Ylist[1:]
            Ylist[-1] = (grad - grad_old)

        else:
            # S = np.hstack([S, x - x_old])
            Slist.append(x - x_old)
            # Y = np.hstack([Y, grad - grad_old])
            Ylist.append(grad - grad_old)

        # 2.
        grad2prev = grad2
        grad2 = pnp.sum(np.asarray(grad) ** 2)

        ######################
        # check if job is done
        # if ((grad2 < g2tol if g2tol is not None else True) and
        #        (pnp.max(np.abs(grad)) < gtol if gtol is not None else
        #        True) and
        #        ((phi - phi_old) / max((1,abs(phi),abs(phi_old))) <= ftol
        #        if ftol is not None else True)):
        callback(x)

        max_grad = pnp.max(np.abs(grad))
        abs_grad = np.linalg.norm(grad)  # TODO[Martin, Lars]: is this parallel ??? # What is this?
        abs_grad = pnp.sum(np.asarray(grad) ** 2) # This is squred norm of gradient
        # max_grad_title = "max ∇ f"
        # abs_grad_title = "|∇ f|"
        phi_change = phi_old - phi

        if disp:
            if MPI.COMM_WORLD.rank == 0:
                print(f"{iteration:<10} {phi:<10.7g} {phi_change:<10.7g} {max_grad:<10.7g} {abs_grad:<10.7g}")

        if (max_grad < gtol):
            print("CONVERGED because gradient tolerance was reached")
            result = OptimizeResult({
                'success': True,
                'x': x.reshape(
                    original_shape),
                'fun': phi,
                'jac': grad.reshape(
                    original_shape),
                'nit': iteration,
                'message':
                    'CONVERGENCE: '
                    'NORM_OF_GRADIENT_<=_GTOL',
                'iterates': iterates,
                'max_grad': max_grad,
                'abs_grad': abs_grad,
                'phi_change': phi_change
            })
            return result

        if (phi_change <= ftol * max((1, abs(phi), abs(phi_old)))):
            print("CONVERGED because function tolerance was reached")
            result = OptimizeResult({
                'success': True,
                'x': x.reshape(
                    original_shape),
                'fun': phi,
                'jac': grad.reshape(
                    original_shape),
                'nit': iteration,
                'message':
                    'CONVERGENCE: '
                    'REL_REDUCTION_OF_F_<=_FACTR*EPSMCH',
                'iterates': iterates,
                'max_grad': max_grad,
                'abs_grad': abs_grad,
                'phi_change': phi_change
            })

            return result

        if iteration > maxiter:
            print("CONVERGED because maximum number of iterations reached")
            result = OptimizeResult({
                'success': False,
                'x': x.reshape(
                    original_shape),
                'fun': phi,
                'jac': grad.reshape(
                    original_shape),
                'nit': iteration,
                'iterates': iterates,
                'max_grad': max_grad,
                'abs_grad': abs_grad,
                'phi_change': phi_change
            })

            return result
        ###########
        # new iteration

        STgrad = np.array([pnp.dot(si.T, grad) for si in Slist]).reshape(-1, 1)
        YTgrad = np.array([pnp.dot(yi.T, grad) for yi in Ylist]).reshape(-1, 1)
        # STgrad = pnp.dot(S.T, grad)
        # YTgrad = pnp.dot(Y.T, grad)

        if iteration > maxcor:
            S_now_T_grad_prev = np.roll(STgrad_prev, -1)
            S_now_T_grad_prev[-1] = - alpha * gamma * grad2prev - alpha * (
                    STgrad_prev.T.dot(p1) + gamma * YTgrad_prev.T.dot(p2))
        else:  # straightforward Version
            S_now_T_grad_prev = np.array(
                [pnp.dot(si.T, grad_old) for si in Slist]).reshape(-1, 1)

        if iteration > maxcor:
            R = np.roll(R, (-1, -1),
                        axis=(0, 1))  # mxm Matrix hold by all Processors
            R[-1, :] = 0
            STym1 = STgrad - S_now_T_grad_prev
            R[:, -1] = STym1.flat  # O(m x n)

        elif iteration == 1:
            # Rm = np.triu(pnp.dot(S.T, Y))
            R = np.triu(np.array(
                [[pnp.dot(si.T, yi).item() for yi in Ylist] for si in Slist]))
            # print(R)
        else:
            # Rm = np.vstack([Rm, np.zeros(k - 1)])
            # Rm = np.hstack([Rm, pnp.dot(S.T, Y[:, -1]).reshape(k, 1)])
            R = np.vstack([R, np.zeros(iteration - 1)])
            R = np.hstack([R, np.array(
                [pnp.dot(si.T, Ylist[-1]) for si in Slist]).reshape(iteration, 1)])
        if iteration > maxcor:
            D = np.roll(D, (-1, -1), axis=(0, 1))
            # D[-1,-1] = np.dot(Y[:,-1],Y[:,-1])# yk-1Tyk-1 # TOOPTIMIZE
            D[-1, -1] = R[-1, -1]
        else:
            # D = np.diag(np.einsum("ik,ik -> k", S, Y))
            D = np.diag(R.diagonal())
        assert D[-1, -1] > 0, "k = {}: ".format(iteration)  # Assumption of Theorem 2.2

        if iteration > maxcor:
            YTY = np.roll(YTY, (-1, -1), axis=(0, 1))
            YTY[-1, :-1] = YTY[:-1, -1] = (YTgrad[:-1] - YTgrad_prev[1:]).flat
            YTY[-1, -1] = grad2prev - grad2 + 2 * YTgrad[-1]
        else:
            # YTYm = pnp.dot(Y.T, Y)
            YTY = np.array(
                [[pnp.dot(yi1.T, yi2).item() for yi2 in Ylist] for yi1 in
                 Ylist])
        # Step 5.
        gamma = D[-1, -1] / YTY[
            -1, -1]  # n.b. D[-1,-1] = sk-1T yk-1 = yk-1T sk-1

        # Step 6. and 7. together
        Rinv = np.linalg.inv(R)
        RiSg = Rinv.dot(STgrad)
        p1 = Rinv.T.dot(D + gamma * YTY).dot(RiSg) - gamma * Rinv.T.dot(YTgrad)
        p2 = - RiSg  # TODO

        # temphstack=np.hstack([S, gamma * Y])

        # Hgradm = gamma * grad + S.dot(p1)  + gamma * Y.dot(p2)
        Hgrad = gamma * grad
        for si, yi, p1i, p2i in zip(Slist, Ylist, p1.flat, p2.flat):
            Hgrad += si * p1i.item() + gamma * yi * p2i.item()

        phi_old = float(phi)
        # printdb("Linesearch: ")
        grad_old[:] = grad
        x_old[:] = x

        def _phi_phiprime(alpha):
            phi, grad[...] = fun_grad(x - Hgrad * alpha, *args)
            phiprime = pnp.dot(grad.T, -Hgrad).item()
            return phi, phiprime

        # TODO: oldphi0: is it allowed to stay outside of the search
        #  direction ?
        alpha, phi, phi0, derphi = scalar_search_wolfe2(
            _phi_phiprime,
            phi0=phi,
            derphi0=pnp.dot(grad.T, -Hgrad).item(),
            maxiter=maxls,
            **linesearch_options)

        printdb("derphi: {}".format(derphi))

        if derphi is None:
            print("line-search did not converge")
            result = OptimizeResult({
                'success': False,
                'x': x.reshape(
                    original_shape),
                'fun': phi,
                'jac': grad.reshape(
                    original_shape),
                'nit': iteration,
                'iterates': iterates,
                'message':
                    'CONVERGENCE: '
                    'line-search did not converge',
                'max_grad': max_grad,
                'abs_grad': abs_grad,
                'phi_change': phi_change
            })
            return result

        x = x - Hgrad * alpha

        if store_iterates:
            iterate = OptimizeResult(
                {
                    'x': x.copy().reshape(original_shape),
                    'fun': phi,
                    'jac': grad.reshape(original_shape),
                    'max_grad': max_grad,
                    'abs_grad': abs_grad,
                    'phi_change': phi_change
                })
            iterates.append(iterate)

        printdb("k = {}".format(iteration))
        iteration = iteration + 1

        STgrad_prev = STgrad.copy()
        YTgrad_prev = YTgrad.copy()
