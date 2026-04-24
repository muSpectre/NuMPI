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
Linear equality constraint utility for constrained optimisation.

Encapsulates a single affine constraint ``<a, x> = target`` together with the
three operations that appear repeatedly in projected-gradient / projected-
quasi-Newton optimisers:

- the Lagrange multiplier ``lambda = <a, grad> / <a, a>`` (closed-form for
  linear equalities),
- the tangent projection of a gradient, ``grad - lambda * a``,
- the Euclidean projection of a point onto the feasible set
  ``F = {x : <a, x> = target, lo <= x <= hi}`` (optionally with a subset of
  indices pinned to zero).

All reductions go through a ``pnp`` object (``NuMPI.Tools.Reduction`` in MPI
mode, ``numpy`` in serial) so the utility is MPI-safe without any
communication code of its own.
"""

import numpy as np


class LinearConstraint:
    r"""
    Single linear equality constraint :math:`\langle a, x \rangle = V^\star`.

    The feasible set (when bounds are supplied to :meth:`project`) is

    .. math::
        F = \bigl\{ x : \langle a, x \rangle = V^\star,\ \mathrm{lo}_i \le x_i \le \mathrm{hi}_i \bigr\}.

    The Euclidean projection :math:`\Pi_F(y)` has the closed form
    :math:`x_i = \mathrm{clip}(y_i - \mu\, a_i,\ \mathrm{lo}_i,\ \mathrm{hi}_i)`
    where the multiplier :math:`\mu` is the unique root of
    :math:`\Phi(\mu) = \sum_i a_i \cdot \mathrm{clip}(y_i - \mu\, a_i,\ \mathrm{lo}_i,\ \mathrm{hi}_i) - V^\star = 0`.
    Under the assumption :math:`a \ge 0` (which holds for volume-like weights
    derived from a non-negative density field), :math:`\Phi` is non-increasing
    in :math:`\mu`, so a simple bracket-and-bisect procedure finds :math:`\mu`
    in a bounded number of MPI reductions.

    Parameters
    ----------
    a : array_like
        Weight vector. Must be non-negative. In MPI-parallel use this is the
        *local* per-rank slice of the global weight vector (same convention
        as ``x`` for the solvers in ``NuMPI.Optimization``).
    target : float
        The right-hand side :math:`V^\star`. **Global** (the same value on
        every rank).
    pnp : module-like, optional
        Reduction wrapper exposing ``sum``. Use ``NuMPI.Tools.Reduction(comm)``
        for MPI parallel runs so that ``<a, a>``, ``<a, grad>`` and the bisection's
        ``Phi(mu)`` sum are taken across ranks; defaults to ``numpy`` for
        serial execution. See the "MPI Conventions" section of the top-level
        README.
    """

    def __init__(self, a, target, pnp=None):
        if pnp is None:
            pnp = np
        self.pnp = pnp
        self.a = np.ascontiguousarray(np.asarray(a).ravel(), dtype=float)
        self.target = float(target)
        self.aTa = float(pnp.sum(self.a * self.a))
        if self.aTa <= 0.0:
            raise ValueError(
                "LinearConstraint: weight vector a is identically zero on all ranks"
            )

    @property
    def size(self):
        """Length of the local portion of the weight vector."""
        return self.a.size

    # ------------------------------------------------------------------
    # Closed-form multiplier and tangent projection of a gradient.
    # ------------------------------------------------------------------

    def multiplier(self, grad, mask=None):
        r"""
        Exact Lagrange multiplier for the linear equality constraint.

        .. math::
            \lambda = \langle a, \mathrm{grad}\rangle / \langle a, a\rangle

        If ``mask`` is a Boolean array, the multiplier is computed with the
        sums restricted to ``mask == True`` (useful when interacting with an
        active-set bound constraint: you typically want :math:`\lambda` taken
        over the *free* indices only).

        Parameters
        ----------
        grad : array_like
        mask : array_like of bool, optional
            Indices to include. ``None`` uses all indices.

        Returns
        -------
        float
        """
        g = np.asarray(grad).ravel()
        if mask is None:
            num = float(self.pnp.sum(self.a * g))
            return num / self.aTa
        m = np.asarray(mask).ravel().astype(bool)
        den = float(self.pnp.sum(np.where(m, self.a * self.a, 0.0)))
        if den == 0.0:
            return 0.0
        num = float(self.pnp.sum(np.where(m, self.a * g, 0.0)))
        return num / den

    def tangent(self, grad, mask=None):
        r"""
        Remove the component of ``grad`` along ``a``: returns
        ``grad - multiplier(grad, mask) * a``.

        The output satisfies :math:`\langle a_{\mathrm{mask}}, \mathrm{out}_{\mathrm{mask}}\rangle = 0`
        (or the unrestricted form when ``mask is None``).
        """
        mu = self.multiplier(grad, mask=mask)
        g = np.asarray(grad)
        return g - mu * self.a.reshape(g.shape)

    # ------------------------------------------------------------------
    # Feasibility projection.
    # ------------------------------------------------------------------

    def project(self, y, lo=None, hi=None, zero_mask=None,
                tol=1e-14, max_iter=60):
        r"""
        Project ``y`` onto the feasible polytope.

        The feasible set is

        .. math::
            F = \{x : \langle a, x\rangle = V^\star,\
                 \mathrm{lo} \le x \le \mathrm{hi},\
                 x_{\mathrm{zero\_mask}} = 0\}.

        Uses bisection on the multiplier :math:`\mu`:
        ``x_i = clip(y_i - mu * a_i, lo_i, hi_i)``, with :math:`\mu` chosen so
        :math:`\langle a, x\rangle = V^\star`.

        Parameters
        ----------
        y : array_like
            The point to project. May violate the bounds or the equality.
        lo, hi : float or array_like, optional
            Lower and upper bounds. ``None`` is treated as :math:`-\infty` or
            :math:`+\infty` respectively.
        zero_mask : array_like of bool, optional
            Indices pinned to ``x = 0`` (e.g. solid-solid contact). Such
            indices are held at zero and do not participate in the linear
            constraint (assumes ``a_i = 0`` at those indices, which is the
            usual case since volume-like weights vanish there).
        tol : float, optional
            Bisection stops early once ``|Phi(mu)| < tol``. Default 1e-14.
        max_iter : int, optional
            Maximum bisection iterations. 60 is sufficient to squeeze any
            practical bracket below Float64 precision.

        Returns
        -------
        ndarray
            Projected point, reshaped to the original shape of ``y``.
        """
        y_arr = np.asarray(y)
        y_flat = y_arr.ravel()
        a = self.a
        pnp = self.pnp
        target = self.target

        lo_arr = np.broadcast_to(np.asarray(-np.inf if lo is None else lo,
                                            dtype=float),
                                 y_flat.shape)
        hi_arr = np.broadcast_to(np.asarray(np.inf if hi is None else hi,
                                            dtype=float),
                                 y_flat.shape)
        if zero_mask is not None:
            zero_mask = np.asarray(zero_mask).ravel().astype(bool)

        def Phi(mu):
            u = np.clip(y_flat - mu * a, lo_arr, hi_arr)
            if zero_mask is not None:
                u = np.where(zero_mask, 0.0, u)
            return float(pnp.sum(a * u)) - target

        # Bracket: Phi(-infty) >= 0, Phi(+infty) <= 0 (a >= 0 makes Phi non-
        # increasing in mu). Starting bracket [-1, 1]; expand geometrically
        # until it straddles zero.
        mu_lo, mu_hi = -1.0, 1.0
        Phi_lo = Phi(mu_lo)
        while Phi_lo < 0.0:
            mu_lo *= 2.0
            if abs(mu_lo) > 1e20:
                raise RuntimeError(
                    "LinearConstraint.project: cannot bracket lower end of mu "
                    "(is the target below Sum(a * lo)?)"
                )
            Phi_lo = Phi(mu_lo)
        Phi_hi = Phi(mu_hi)
        while Phi_hi > 0.0:
            mu_hi *= 2.0
            if abs(mu_hi) > 1e20:
                raise RuntimeError(
                    "LinearConstraint.project: cannot bracket upper end of mu "
                    "(is the target above Sum(a * hi)?)"
                )
            Phi_hi = Phi(mu_hi)

        mu = 0.5 * (mu_lo + mu_hi)
        for _ in range(max_iter):
            mu_mid = 0.5 * (mu_lo + mu_hi)
            Phi_mid = Phi(mu_mid)
            if abs(Phi_mid) < tol:
                # Use mu_mid itself — the bracket average can straddle the
                # root and land far from it when the root is hit exactly
                # (e.g. if Phi(0) == 0 on the first iteration).
                mu = mu_mid
                break
            if Phi_mid > 0.0:
                mu_lo = mu_mid
            else:
                mu_hi = mu_mid
            mu = 0.5 * (mu_lo + mu_hi)

        x = np.clip(y_flat - mu * a, lo_arr, hi_arr)
        if zero_mask is not None:
            x = np.where(zero_mask, 0.0, x)
        return x.reshape(y_arr.shape)
