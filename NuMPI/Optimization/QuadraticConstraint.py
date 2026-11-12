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
``BilinearConstraint`` is the special case ``sum_i w_i g_i h_i = target`` of
two fields ``x = (g, h)``, for which the restoration is the exact Euclidean
projection.

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


def _movable(raw, lo, hi, zero, push=None):
    """
    Mask of the entries of ``raw`` that follow a move along the normals.

    These are the entries strictly inside their bounds and, if ``push`` is
    given, the entries exactly on one bound that ``push`` moves into the box.

    Args:
        raw: Local slice of the unclipped point.
        lo, hi: Bounds, scalar, array or ``None``.
        zero: Boolean array of the entries that never move, or ``None``.
        push: Local array of the signed displacement the move is going to
            apply, or ``None``.

    Returns:
        Boolean array shaped like ``raw``.
    """
    above = np.ones(raw.shape, dtype=bool) if lo is None else raw > lo
    below = np.ones(raw.shape, dtype=bool) if hi is None else raw < hi
    movable = above & below
    if push is not None:
        if lo is not None:
            movable |= (raw == lo) & below & (push > 0.0)
        if hi is not None:
            movable |= (raw == hi) & above & (push < 0.0)
    if zero is not None:
        movable &= ~zero
    return movable


def _quadratic_root(r, J, q):
    """
    Root of ``r + J t + q t^2 / 2`` nearest to ``t = 0``.

    Returns:
        The root, or ``None`` if there is no real root or ``J`` vanishes.
    """
    if not (np.isfinite(r) and np.isfinite(J) and np.isfinite(q)) or J == 0.0:
        return None
    if q == 0.0:
        return -r / J
    disc = J * J - 2.0 * q * r
    if disc < 0.0:
        return None
    return -2.0 * r / (J + np.copysign(np.sqrt(disc), J))


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
        pnp: Reduction wrapper exposing ``sum`` and ``min``. ``None`` means
            ``numpy``.

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

    def products(self, u):
        """
        The products :math:`B_j u`.

        Args:
            u: Flat local slice of a vector.

        Returns:
            Local ``(m, n)`` array whose row ``j`` is :math:`B_j u`.
        """
        if self.hessp is None:
            return np.zeros((self.m, u.size))
        return np.asarray(self.hessp(u), dtype=float).reshape(self.m, u.size)

    def scaled(self, scale):
        r"""
        The same constraints in the variables :math:`u = x / s`.

        Args:
            scale: Local slice of the positive scale :math:`s`.

        Returns:
            ``QuadraticConstraint`` with linear terms :math:`s a_j` and
            products :math:`s\, B_j (s u)`.
        """
        scale = np.asarray(scale, dtype=float).ravel()

        def hessp(u):
            return scale * self.products(scale * u)

        return QuadraticConstraint(
            self.a * scale, None if self.hessp is None else hessp, self.target,
            pnp=self.pnp,
        )

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
        Bu, Bv = self.products(u), self.products(v)
        defect = self.products(u - 2.0 * v) - Bu + 2.0 * Bv
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
        Bx = self.products(x)
        return _reduce(0.5 * (Bx @ x) + self.a @ x, self.pnp)

    def normals(self, x, products=None):
        """
        The normals :math:`B_j x + a_j`.

        Args:
            x: Flat local slice of the point.
            products: The products :math:`B_j x` as returned by
                :meth:`products`, if they are known. ``None`` computes them.

        Returns:
            Local ``(n, m)`` array whose column ``j`` is the normal of
            component ``j``.
        """
        if products is None:
            products = self.products(np.asarray(x, dtype=float).ravel())
        return (products + self.a).T

    def multiplier(self, grad, x, mask=None, normals=None):
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
            normals: The normals at ``x`` as returned by :meth:`normals`, if
                they are known. ``None`` computes them.

        Returns:
            Array of length ``m``, or ``(r, m)`` for a stack.

        Raises:
            numpy.linalg.LinAlgError: The Gram matrix of the normals on
                ``mask`` is singular.
        """
        N = self.normals(x) if normals is None else normals
        return self._fit(np.asarray(grad, dtype=float), N, mask)

    def tangent(self, grad, x, mask=None, return_multiplier=False, normals=None):
        r"""
        ``grad`` with its components along the normals at ``x`` removed,
        :math:`\mathrm{grad} - N \lambda` with :math:`\lambda` from
        :meth:`multiplier`.

        Args:
            grad: Flat local vector, or an ``(r, n)`` stack of them.
            x: Flat local slice of the point.
            mask: Boolean array of the entries in the fit. ``None`` uses all.
            return_multiplier: Also return :math:`\lambda`.
            normals: The normals at ``x`` as returned by :meth:`normals`, if
                they are known. ``None`` computes them.

        Returns:
            The tangent vector, shaped like ``grad``, and with
            ``return_multiplier`` also the multipliers.

        Raises:
            numpy.linalg.LinAlgError: The Gram matrix of the normals on
                ``mask`` is singular.
        """
        grad = np.asarray(grad, dtype=float)
        N = self.normals(x) if normals is None else normals
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

    def project(self, y, lo=None, hi=None, zero_mask=None, max_rounds=20,
                max_newton=30, fixed=None):
        """
        Restore ``y`` onto the constraint set, see :meth:`restore`.

        Returns:
            The restored point, or ``None`` if the restoration fails.
        """
        restored = self.restore(
            y, lo=lo, hi=hi, zero_mask=zero_mask, max_rounds=max_rounds,
            max_newton=max_newton, fixed=fixed,
        )
        return None if restored is None else restored[0]

    def restore(self, y, lo=None, hi=None, zero_mask=None, max_rounds=20,
                max_newton=30, products=None, fixed=None):
        r"""
        Restore ``y`` onto the constraint set along the clipped normal path.

        With :math:`\hat y` the point ``y`` clipped to the bounds and
        :math:`N = N(\hat y)` its normals, the restored point is

        .. math::
            x(t) = \mathrm{clip}(y + N t,\ \mathrm{lo},\ \mathrm{hi}),
            \qquad c_j(x(t)) = c_j^*,

        with :math:`t \in \mathbb{R}^m`. On the entries that are not clipped,
        the constraints are polynomials of degree two in :math:`t`; their root
        gives a new :math:`t`, from which the set of clipped entries is
        updated, until it no longer changes. For one constraint the root is
        in closed form, and once two :math:`t` of opposite residual are known
        the iteration is safeguarded by bisection. For several constraints
        the root is found by Newton's method.

        For a linear constraint this is the Euclidean projection onto the
        feasible set.

        Args:
            y: Flat local slice of the point.
            lo, hi: Bounds, scalar, array or ``None``.
            zero_mask: Boolean array of the entries pinned to zero.
            max_rounds: Maximum number of updates of the clipped set.
            max_newton: Maximum Newton iterations per update, for more than
                one constraint.
            products: The products :math:`B_j y` as returned by
                :meth:`products`, if they are known. They are used if
                clipping does not change ``y``.
            fixed: Boolean array of the entries that keep their clipped
                value: the move along the normals leaves them alone. An
                optimiser passes the entries it holds on their bounds, which
                a curved constraint would otherwise pull into the box by a
                small amount at every step.

        Returns:
            ``(x, products)``, the restored point and the products
            :math:`B_j x`, or ``None`` if no restoration satisfies every
            component within ``max_rounds``.
        """
        m = self.m
        pnp = self.pnp
        y = np.asarray(y, dtype=float).ravel()
        zero = None if zero_mask is None else np.asarray(zero_mask, dtype=bool).ravel()
        y0 = y if zero is None else np.where(zero, 0.0, y)

        def clip(raw):
            x = _clip(raw, lo, hi)
            return x if zero is None else np.where(zero, 0.0, x)

        pinned = zero
        if fixed is not None:
            fixed = np.asarray(fixed, dtype=bool).ravel()
            pinned = fixed if zero is None else zero | fixed

        x = clip(y0)
        if products is not None and not (lo is None and hi is None and zero is None):
            if _reduce([np.count_nonzero(x != y)], pnp)[0] != 0.0:
                products = None
        Bx = self.products(x) if products is None else np.asarray(products)
        # The direction of the move stays the normals at the clipped point.
        N = (Bx + self.a).T
        if pinned is not None:
            N = np.where(pinned[:, np.newaxis], 0.0, N)

        push = None
        if m == 1 and not (lo is None and hi is None):
            # The first move has the sign of -r0, so it is known which entries
            # on a bound it carries into the box.
            r0 = _reduce([0.5 * (Bx[0] @ x) + self.a[0] @ x], pnp)[0] - self.target[0]
            push = -np.sign(r0) * N[:, 0]

        t = np.zeros(m)
        raw = y0
        t_pos = t_neg = None
        for round_ in range(max_rounds):
            if round_ > 0:
                Bx = self.products(x)
            free = _movable(raw, lo, hi, pinned, push)
            push = None
            NF = np.where(free[:, np.newaxis], N, 0.0)
            # Row j of BNF[i] is B_j applied to free normal i.
            BNF = np.stack([self.products(NF[:, i]) for i in range(m)])
            total = _reduce(
                np.concatenate([
                    0.5 * (Bx @ x) + self.a @ x,
                    ((Bx + self.a) @ NF).ravel(),
                    np.einsum("ijn,nk->jik", BNF, NF).ravel(),
                ]),
                pnp,
            )
            r0 = total[:m] - self.target
            J0 = total[m:m + m * m].reshape(m, m)
            Q = total[m + m * m:].reshape(m, m, m)
            scale = np.abs(self.target) + np.abs(r0) + np.finfo(float).tiny
            if not np.all(np.isfinite(total)):
                return None
            if np.all(np.abs(r0) <= 1e-14 * scale):
                return x, Bx

            # Root of r_j(d) = r0_j + J0_j . d + d^T Q_j d / 2 nearest to 0;
            # x + NF d satisfies the constraints if no entry changes sides.
            exact = True
            if m == 1:
                delta = _quadratic_root(r0[0], J0[0, 0], Q[0, 0, 0])
                if r0[0] > 0.0:
                    t_pos = t[0]
                else:
                    t_neg = t[0]
                if t_pos is not None and t_neg is not None:
                    low, high = min(t_pos, t_neg), max(t_pos, t_neg)
                    if delta is None or not low < t[0] + delta < high:
                        delta, exact = 0.5 * (low + high) - t[0], False
                elif delta is None:
                    # No entry follows the move, or the model has no root:
                    # go on to where the next clipped entry enters the box.
                    delta, exact = self._next_entry(raw, N[:, 0], lo, hi, free, -r0[0]), False
                delta = None if delta is None else np.array([delta])
            else:
                delta = self._newton(r0, J0, Q, max_newton)
            if delta is None:
                return None

            t = t + delta
            raw = y0 + N @ t
            x_new = clip(raw)
            changed = (_movable(raw, lo, hi, pinned) != free) | (~free & (x_new != x))
            if exact and _reduce([np.count_nonzero(changed)], pnp)[0] == 0.0:
                return x_new, Bx + np.einsum("i,ijn->jn", delta, BNF)
            x = x_new
        return None

    def _next_entry(self, raw, normal, lo, hi, free, direction):
        # Smallest step of t, of the sign of `direction`, that carries an
        # entry clipped at a bound just into the box; None if there is none.
        gap = np.zeros(raw.shape)
        if lo is not None:
            gap = np.where(raw < lo, lo - raw, gap)
        if hi is not None:
            gap = np.where(raw > hi, hi - raw, gap)
        with np.errstate(divide="ignore", invalid="ignore"):
            step = np.where(~free & (gap != 0.0) & (normal != 0.0), gap / normal, np.inf)
        step = np.where(step * direction > 0.0, np.abs(step), np.inf)
        # A Reduction returns the largest float, not inf, if there is none.
        nearest = float(self.pnp.min(step, initial=np.inf))
        if not nearest < np.finfo(float).max:
            return None
        return np.sign(direction) * nearest * (1.0 + 1e-8)

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


def _node_minimisers(yg, yh, nu, lg, ug, lh, uh):
    """
    Minimisers of ``(g - yg)^2 / 2 + (h - yh)^2 / 2 + nu g h`` over the boxes
    ``lg <= g <= ug``, ``lh <= h <= uh``, node by node, for ``|nu| < 1``.

    The function is strictly convex. Its minimiser is the stationary point if
    that lies in the box, and otherwise the best of the minimisers along the
    four edges.

    Returns:
        ``(g, h)``.
    """
    den = 1.0 - nu * nu
    g = (yg - nu * yh) / den
    h = (yh - nu * yg) / den
    outside = (g < lg) | (g > ug) | (h < lh) | (h > uh)
    if not np.any(outside):
        return g, h
    best = np.full(yg.shape, np.inf)
    g_edge, h_edge = g.copy(), h.copy()
    for bound in (lg, ug):
        valid = outside & np.isfinite(bound)
        gc = np.where(valid, bound, 0.0)
        hc = np.clip(yh - nu * gc, lh, uh)
        q = np.where(valid, 0.5 * (gc - yg) ** 2 + 0.5 * (hc - yh) ** 2 + nu * gc * hc, np.inf)
        better = q < best
        best, g_edge, h_edge = np.where(better, q, best), np.where(better, gc, g_edge), np.where(better, hc, h_edge)
    for bound in (lh, uh):
        valid = outside & np.isfinite(bound)
        hc = np.where(valid, bound, 0.0)
        gc = np.clip(yg - nu * hc, lg, ug)
        q = np.where(valid, 0.5 * (gc - yg) ** 2 + 0.5 * (hc - yh) ** 2 + nu * gc * hc, np.inf)
        better = q < best
        best, g_edge, h_edge = np.where(better, q, best), np.where(better, gc, g_edge), np.where(better, hc, h_edge)
    return np.where(outside, g_edge, g), np.where(outside, h_edge, h)


class BilinearConstraint(QuadraticConstraint):
    r"""
    Bilinear equality constraint
    :math:`\sum_i w_i g_i h_i = V` of two fields :math:`x = (g, h)`.

    This is the :class:`QuadraticConstraint` with :math:`a = 0` and
    :math:`B (u_g, u_h) = (w u_h, w u_g)`. Because :math:`B` couples only the
    two unknowns of one node, :meth:`restore` is the exact Euclidean
    projection onto the feasible set, including the bounds.

    Args:
        n: Local number of nodes. The local slice of ``x`` is the ``n``
            entries of :math:`g` followed by the ``n`` entries of :math:`h`.
        target: The right-hand side :math:`V`, the same on every rank.
        weights: Local slice of the weights :math:`w`, scalar or array of
            length ``n``. ``None`` means 1.
        pnp: Reduction wrapper exposing ``sum`` and ``max``. ``None`` means
            ``numpy``.
    """

    def __init__(self, n, target, weights=None, pnp=None):
        self.n = int(n)
        self.weights = np.broadcast_to(
            np.asarray(1.0 if weights is None else weights, dtype=float), (self.n,)
        ).copy()

        def hessp(u):
            return np.concatenate([self.weights * u[self.n:], self.weights * u[:self.n]])

        super().__init__(np.zeros(2 * self.n), hessp, target, pnp=pnp)
        self.max_weight = float(self.pnp.max(np.abs(self.weights), initial=0.0))
        if not np.isfinite(self.max_weight):
            raise ValueError("BilinearConstraint: the weights are not finite")

    def scaled(self, scale):
        r"""
        The same constraint in the variables :math:`u = x / s`.

        Args:
            scale: Local slice of the positive scale :math:`s`, length ``2 n``.

        Returns:
            ``BilinearConstraint`` with weights :math:`w\, s_g s_h`.
        """
        scale = np.broadcast_to(np.asarray(scale, dtype=float).ravel(), (2 * self.n,))
        return BilinearConstraint(
            self.n, self.target[0],
            weights=self.weights * scale[:self.n] * scale[self.n:], pnp=self.pnp,
        )

    def restore(self, y, lo=None, hi=None, zero_mask=None, max_rounds=20,
                max_newton=30, products=None, fixed=None, max_iter=100):
        r"""
        Euclidean projection of ``y`` onto the feasible set.

        The projection minimises :math:`\tfrac12 \lVert x - y\rVert^2` on the
        constraint and the bounds. With the multiplier :math:`\mu` of the
        constraint, the Lagrangian is a sum over the nodes of

        .. math::
            \tfrac12 (g_i - y^g_i)^2 + \tfrac12 (h_i - y^h_i)^2
            + \mu\, w_i\, g_i h_i ,

        which is strictly convex for :math:`\lvert\mu w_i\rvert < 1` and is
        minimised over the box of node :math:`i` in closed form. The residual
        :math:`\Phi(\mu) = \sum_i w_i g_i(\mu) h_i(\mu) - V` of these
        minimisers does not increase with :math:`\mu`, and its root in
        :math:`\lvert\mu\rvert < 1 / \max_i \lvert w_i\rvert` gives the
        projection. The root is bracketed and found by the Illinois variant
        of the regula falsi, with one reduction per evaluation and no call of
        ``hessp``.

        If :math:`\Phi` has no root in that interval, the projection is not
        the minimiser of a convex problem, and the restoration of
        :meth:`QuadraticConstraint.restore` is returned instead.

        Args:
            y: Flat local slice of the point, :math:`g` followed by :math:`h`.
            lo, hi: Bounds, scalar, array or ``None``.
            zero_mask: Boolean array of the entries pinned to zero.
            fixed: Boolean array of the entries that keep their clipped
                value; the projection is onto the feasible points that agree
                with ``y`` there.
            max_rounds, max_newton, products: Passed to
                :meth:`QuadraticConstraint.restore` if it is used.
            max_iter: Maximum number of evaluations of :math:`\Phi`.

        Returns:
            ``(x, products)``, the projected point and the product
            :math:`B x`, or ``None`` if the fallback fails.
        """
        n = self.n
        w = self.weights
        pnp = self.pnp
        target = self.target[0]
        y = np.asarray(y, dtype=float).ravel()

        def fallback(start):
            return super(BilinearConstraint, self).restore(
                start, lo=lo, hi=hi, zero_mask=zero_mask, max_rounds=max_rounds,
                max_newton=max_newton, fixed=fixed,
                products=products if start is y else None,
            )

        if not self.max_weight > 0.0:
            return fallback(y)

        lo_arr = np.broadcast_to(
            np.asarray(-np.inf if lo is None else lo, dtype=float), y.shape)
        hi_arr = np.broadcast_to(
            np.asarray(np.inf if hi is None else hi, dtype=float), y.shape)
        if zero_mask is not None:
            zero = np.asarray(zero_mask, dtype=bool).ravel()
            lo_arr = np.where(zero, 0.0, lo_arr)
            hi_arr = np.where(zero, 0.0, hi_arr)
        if fixed is not None:
            fixed = np.asarray(fixed, dtype=bool).ravel()
            kept = np.clip(y, lo_arr, hi_arr)
            lo_arr = np.where(fixed, kept, lo_arr)
            hi_arr = np.where(fixed, kept, hi_arr)
        boxes = (lo_arr[:n], hi_arr[:n], lo_arr[n:], hi_arr[n:])

        def residual(mu):
            g, h = _node_minimisers(y[:n], y[n:], mu * w, *boxes)
            return _reduce([np.sum(w * g * h)], pnp)[0] - target, np.concatenate([g, h])

        def done(x):
            return x, self.products(x)

        f0, x0 = residual(0.0)
        if not np.isfinite(f0):
            return None
        tol = 1e-14 * (abs(target) + abs(f0) + np.finfo(float).tiny)
        if abs(f0) <= tol:
            return done(x0)

        # Phi does not increase with mu: the root is at mu > 0 if Phi(0) > 0.
        # Approach the end of the convex interval geometrically.
        cap = np.copysign((1.0 - 1e-9) / self.max_weight, f0)
        a, fa = 0.0, f0
        for k in range(1, 31):
            b = cap * (1.0 - 0.5 ** k)
            fb, xb = residual(b)
            if not np.isfinite(fb):
                return fallback(y)
            if fb * f0 <= 0.0:
                break
            a, fa = b, fb
        else:
            return fallback(y)

        for _ in range(max_iter):
            if abs(fb) <= tol or abs(b - a) <= 4.0 * np.finfo(float).eps * max(abs(a), abs(b)):
                break
            c = b - fb * (b - a) / (fb - fa)
            fc, xc = residual(c)
            if fc * fb < 0.0:
                a, fa = b, fb
            else:
                fa *= 0.5
            b, fb, xb = c, fc, xc
        if abs(fb) <= 1e4 * tol:
            return done(xb)
        # The sum cannot be resolved to the tolerance; finish with a
        # restoration from the nearly feasible point.
        return fallback(xb)
