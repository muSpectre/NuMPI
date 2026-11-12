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
Quadratic equality constraints

    c_j(x) = x^T B_j x / 2 + <a_j, x> = target_j,    j = 1, ..., m,

with constant symmetric matrices ``B_j``, together with the joint multiplier,
tangent projection and restoration onto the constraint set.

All reductions go through a ``pnp`` object (``NuMPI.Tools.Reduction`` in MPI
mode, ``numpy`` in serial).
"""

import numpy as np

_TOL_BOX = 1e-12
_MAX_GRAM_COND = 1e12


def _reduce(local, pnp):
    """
    Element-wise global sum of a local array, in one reduction.

    Args:
        local: Local array of per-rank partial sums.
        pnp: Reduction wrapper.

    Returns:
        Array of the shape of ``local`` holding the global sums.
    """
    local = np.asarray(local, dtype=float)
    return np.asarray(
        pnp.sum(local.reshape(1, -1), axis=0), dtype=float
    ).reshape(local.shape)


def _clip(x, lo, hi):
    if lo is not None:
        x = np.maximum(x, lo)
    if hi is not None:
        x = np.minimum(x, hi)
    return x


def _active(x, lo, hi, zero_mask):
    """
    Masks of the entries on the lower bound, on the upper bound, and in
    ``zero_mask``.

    Args:
        x: Local slice of the point.
        lo, hi: Bounds, scalar, array or ``None``.
        zero_mask: Boolean array or ``None``.

    Returns:
        ``(at_lo, at_hi, zero)``, three boolean arrays shaped like ``x``.
    """
    none = np.zeros(x.shape, dtype=bool)
    at_lo = none if lo is None else x <= np.asarray(lo) + _TOL_BOX
    at_hi = none if hi is None else x >= np.asarray(hi) - _TOL_BOX
    zero = none if zero_mask is None else np.asarray(zero_mask, dtype=bool)
    return at_lo, at_hi, zero


class QuadraticConstraint:
    r"""
    Quadratic equality constraints
    :math:`c_j(x) = \tfrac12 x^{\mathsf T} B_j x + \langle a_j, x\rangle = c_j^*`,
    :math:`j = 1, \dots, m`.

    The normal of component :math:`j` is :math:`B_j x + a_j`.

    Args:
        a: Local slice of the linear terms, an ``(m, n)`` array whose row
            ``j`` is :math:`a_j`. A 1-D array means ``m = 1``.
        hessp: ``hessp(u)`` returns, for the flat local slice ``u``, the
            local ``(m, n)`` array whose row ``j`` is :math:`B_j u`. It may
            exchange ghost values, so every rank calls it together. It must
            be linear in ``u`` and symmetric, see :meth:`check_hessp`.
            ``None`` means every :math:`B_j = 0`.
        target: The right-hand sides :math:`c_j^*`, length ``m``, the same on
            every rank.
        pnp: Reduction wrapper exposing ``sum``. ``None`` means ``numpy``.

    Raises:
        ValueError: ``target`` does not have one entry per row of ``a``.
    """

    def __init__(self, a, hessp, target, pnp=None):
        if pnp is None:
            pnp = np
        self.pnp = pnp
        a = np.asarray(a, dtype=float)
        if a.ndim == 1:
            a = a[np.newaxis, :]
        self.a = np.ascontiguousarray(a)
        self.hessp = hessp
        self.target = np.atleast_1d(np.asarray(target, dtype=float)).ravel()
        if self.target.size != self.a.shape[0]:
            raise ValueError(
                f"QuadraticConstraint: target has {self.target.size} entries, "
                f"a has {self.a.shape[0]} rows"
            )

    @property
    def m(self):
        """Number of components."""
        return self.a.shape[0]

    def _products(self, u):
        if self.hessp is None:
            return np.zeros((self.m, u.size))
        return np.asarray(self.hessp(u), dtype=float).reshape(self.m, u.size)

    def check_hessp(self, rtol=1e-8, rng=None):
        r"""
        Test that ``hessp`` is a symmetric linear map.

        The value :math:`\tfrac12 x^{\mathsf T} B_j x + \langle a_j, x\rangle`,
        the normal :math:`B_j x + a_j` and the restoration in :meth:`project`
        are exact only for constant symmetric :math:`B_j`. Two random probes
        :math:`u, v` test linearity,
        :math:`B_j (u - 2 v) = B_j u - 2 B_j v`, and symmetry,
        :math:`\langle v, B_j u\rangle = \langle u, B_j v\rangle`. This
        takes three calls of ``hessp`` and one reduction.

        A product that is not linear in its argument (a Hessian that depends
        on the point, an affine map) or that represents the quadratic form by
        a non-symmetric matrix fails the test.

        Args:
            rtol: Relative tolerance of both tests.
            rng: ``numpy.random.Generator`` of the probes. ``None`` draws
                them from fresh entropy.

        Raises:
            ValueError: ``hessp`` is not linear or not symmetric.
        """
        if self.hessp is None:
            return
        if rng is None:
            rng = np.random.default_rng()
        u, v = rng.standard_normal((2, self.a.shape[1]))
        Bu, Bv = self._products(u), self._products(v)
        defect = self._products(u - 2.0 * v) - Bu + 2.0 * Bv
        m = self.m
        total = _reduce(
            np.concatenate([
                np.sum(defect * defect, axis=1),
                np.sum(Bu * Bu, axis=1),
                np.sum(Bv * Bv, axis=1),
                Bu @ v,
                Bv @ u,
                [u @ u, v @ v],
            ]),
            self.pnp,
        )
        defect2, BuBu, BvBv, vBu, uBv = np.split(total[:5 * m], 5)
        uu, vv = total[5 * m:]
        nonlinear = np.sqrt(defect2) > rtol * np.sqrt(BuBu + 4.0 * BvBv)
        if np.any(nonlinear) or not np.all(np.isfinite(total)):
            raise ValueError(
                "QuadraticConstraint: hessp is not linear in its argument for "
                f"component(s) {np.flatnonzero(nonlinear).tolist()}; the "
                "constraint must be quadratic with a constant Hessian"
            )
        asymmetric = np.abs(vBu - uBv) > rtol * (
            np.sqrt(vv * BuBu) + np.sqrt(uu * BvBv)
        )
        if np.any(asymmetric):
            raise ValueError(
                "QuadraticConstraint: hessp is not symmetric for component(s) "
                f"{np.flatnonzero(asymmetric).tolist()}; pass the product with "
                "the symmetric part (B + B^T) / 2 of the matrix"
            )

    def value(self, x):
        """
        The constraint values :math:`c_j(x)`.

        Args:
            x: Flat local slice of the point.

        Returns:
            Array of length ``m``, the same on every rank.
        """
        x = np.asarray(x, dtype=float).ravel()
        Bx = self._products(x)
        return _reduce(0.5 * (Bx @ x) + self.a @ x, self.pnp)

    def normals(self, x):
        """
        The normals :math:`B_j x + a_j`.

        Args:
            x: Flat local slice of the point.

        Returns:
            Local ``(n, m)`` array whose column ``j`` is the normal of
            component ``j``.
        """
        x = np.asarray(x, dtype=float).ravel()
        return (self._products(x) + self.a).T

    def multiplier(self, grad, x, mask=None):
        r"""
        Least-squares multipliers of ``grad`` in the normals at ``x``,

        .. math::
            \lambda = (N_{\mathrm{mask}}^{\mathsf T} N_{\mathrm{mask}})^{-1}
                      N_{\mathrm{mask}}^{\mathsf T}\, \mathrm{grad},

        with every sum taken over ``mask``.

        Args:
            grad: Flat local vector, or an ``(r, n)`` stack of them.
            x: Flat local slice of the point.
            mask: Boolean array of the entries in the fit. ``None`` uses all.

        Returns:
            Array of length ``m``, or ``(r, m)`` for a stack.

        Raises:
            numpy.linalg.LinAlgError: The Gram matrix of the normals on
                ``mask`` is singular.
        """
        return self._fit(np.asarray(grad, dtype=float), self.normals(x), mask)

    def tangent(self, grad, x, mask=None, return_multiplier=False):
        r"""
        ``grad`` with its components along the normals at ``x`` removed,
        :math:`\mathrm{grad} - N \lambda` with :math:`\lambda` from
        :meth:`multiplier`.

        Args:
            grad: Flat local vector, or an ``(r, n)`` stack of them.
            x: Flat local slice of the point.
            mask: Boolean array of the entries in the fit. ``None`` uses all.
            return_multiplier: Also return :math:`\lambda`.

        Returns:
            The tangent vector, shaped like ``grad``, and with
            ``return_multiplier`` also the multipliers.

        Raises:
            numpy.linalg.LinAlgError: The Gram matrix of the normals on
                ``mask`` is singular.
        """
        grad = np.asarray(grad, dtype=float)
        N = self.normals(x)
        lam = self._fit(grad, N, mask)
        out = grad - lam @ N.T
        return (out, lam) if return_multiplier else out

    def _fit(self, grad, N, mask):
        if mask is None:
            mask = np.ones(N.shape[0], dtype=bool)
        m = self.m
        rhs = np.atleast_2d(grad)
        Nm = N[mask]
        gram, b = np.split(
            _reduce(np.concatenate([(Nm.T @ Nm).ravel(), (rhs[:, mask] @ Nm).ravel()]),
                    self.pnp),
            [m * m],
        )
        gram = gram.reshape(m, m)
        cond = np.linalg.cond(gram)
        if not cond <= _MAX_GRAM_COND:
            raise np.linalg.LinAlgError(
                "QuadraticConstraint: the Gram matrix of the normals on the mask "
                f"is singular, condition number {cond:.3e}"
            )
        lam = np.linalg.solve(gram, b.reshape(-1, m).T).T
        return lam[0] if np.ndim(grad) == 1 else lam

    def project(self, y, lo=None, hi=None, zero_mask=None, max_rounds=10,
                max_newton=30):
        r"""
        Restore ``y`` onto the constraint set.

        ``y`` is clipped to the bounds and moved along the normals at the
        clipped point, masked to the entries strictly inside their bounds:
        :math:`x = \tilde y + N_F t`. The step sizes :math:`t \in \mathbb{R}^m`
        solve the polynomial system :math:`c_j(\tilde y + N_F t) = c_j^*` by
        Newton's method from :math:`t = 0`. An entry that crosses its bound
        is clipped and removed from the free entries, and the restoration is
        repeated.

        Args:
            y: Flat local slice of the point.
            lo, hi: Bounds, scalar, array or ``None``.
            zero_mask: Boolean array of the entries pinned to zero.
            max_rounds: Maximum number of restorations.
            max_newton: Maximum Newton iterations per restoration.

        Returns:
            The restored point, or ``None`` if no restoration satisfies every
            component within ``max_rounds``.
        """
        m = self.m
        x = _clip(np.asarray(y, dtype=float).ravel(), lo, hi)
        if zero_mask is not None:
            x = np.where(zero_mask, 0.0, x)
        at_lo, at_hi, zero = _active(x, lo, hi, zero_mask)
        free = ~(at_lo | at_hi | zero)
        for _ in range(max_rounds):
            Bx = self._products(x)
            NF = np.where(free[:, np.newaxis], (Bx + self.a).T, 0.0)
            # Row j of BNF[i] is B_j applied to free normal i.
            BNF = np.stack([self._products(NF[:, i]) for i in range(m)])
            total = _reduce(
                np.concatenate([
                    0.5 * (Bx @ x) + self.a @ x,
                    (NF.T @ NF).ravel(),
                    np.einsum("ijn,nk->jik", BNF, NF).ravel(),
                ]),
                self.pnp,
            )
            c0 = total[:m] - self.target
            J0 = total[m:m + m * m].reshape(m, m)
            Q = total[m + m * m:].reshape(m, m, m)

            t = self._newton(c0, J0, Q, max_newton)
            if t is None:
                return None
            x_new = x + NF @ t
            crossed = np.zeros(x.shape, dtype=bool)
            if lo is not None:
                crossed |= x_new < lo
            if hi is not None:
                crossed |= x_new > hi
            crossed &= free
            if float(self.pnp.sum(crossed.astype(float))) == 0.0:
                return x_new
            x = _clip(x_new, lo, hi)
            free &= ~crossed
        return None

    def _newton(self, c0, J0, Q, max_newton):
        # Root of r_j(t) = c0_j + J0_j . t + t^T Q_j t / 2 nearest to t = 0.
        scale = np.abs(self.target) + np.abs(c0) + np.finfo(float).tiny
        t = np.zeros_like(c0)
        for _ in range(max_newton):
            r = c0 + J0 @ t + 0.5 * np.einsum("i,jik,k->j", t, Q, t)
            if np.all(np.abs(r) <= 1e-14 * scale):
                return t
            J = J0 + Q @ t
            if not np.all(np.isfinite(J)) or not np.linalg.cond(J) <= _MAX_GRAM_COND:
                return None
            t = t - np.linalg.solve(J, r)
        r = c0 + J0 @ t + 0.5 * np.einsum("i,jik,k->j", t, Q, t)
        return t if np.all(np.abs(r) <= 1e-10 * scale) else None
