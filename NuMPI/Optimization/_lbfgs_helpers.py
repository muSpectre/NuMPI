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
Shared helpers for the projected / bound-constrained L-BFGS solvers.

Kept generic: the functions operate on plain arrays and only touch MPI
through the caller-supplied ``pnp`` reduction wrapper.
"""

import numpy as np


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


def _kkt_residual(grad, x, lo, hi, zero_mask=None, tol_box=1e-12):
    """
    Box-active-masked gradient. Components where ``x`` sits on an active box
    face *and* ``grad`` points into that face are zeroed (complementary
    slackness satisfied, no improvement possible). Indices in ``zero_mask``
    are always zeroed.

    The first argument may be either a raw gradient (bound-constrained case)
    or a tangent-projected gradient (linear-equality + box case); the mask
    logic is the same.
    """
    r = grad.copy()
    if zero_mask is not None:
        r = np.where(zero_mask, 0.0, r)
    if lo is not None:
        r = np.where((x <= lo + tol_box) & (grad >= 0.0), 0.0, r)
    if hi is not None:
        r = np.where((x >= hi - tol_box) & (grad <= 0.0), 0.0, r)
    return r


def _armijo(fun_grad, x, d, phi, gd, project, *, c1, max_halvings, alpha_init):
    """
    Projected backtracking line search. Returns
    ``(x_new, phi_new, grad_new, alpha)`` on success, ``None`` on failure.
    """
    alpha = alpha_init
    for _ in range(max_halvings):
        x_try = project(x + alpha * d)
        phi_try, grad_try = fun_grad(x_try)
        if phi_try <= phi + c1 * alpha * gd:
            return x_try, phi_try, grad_try, alpha
        alpha *= 0.5
    return None
