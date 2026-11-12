# Optimizers

```python
from NuMPI.Optimization import (
    l_bfgs, l_bfgs_bounded, l_bfgs_projected, l_bfgs_qc_projected,
    tr_newton_bounded,
    constrained_conjugate_gradients, constrained_conjugate_gradients_with_restart,
    LinearConstraint, QuadraticConstraint,
)
```

NuMPI's minimizers operate on distributed arrays: every process holds a slice
of the unknowns $x$, and all inner products, norms and convergence tests are
reduced over the communicator. Without a communicator they run serially.

| Optimizer | Problem | Needs |
|-----------|---------|-------|
| `l_bfgs` | $\min f(x)$ | $f$, $\nabla f$ |
| `l_bfgs_bounded` | $\min f(x)$ subject to $l \le x \le h$ | $f$, $\nabla f$ |
| `l_bfgs_projected` | $\min f(x)$ subject to $\langle a, x\rangle = V^\star$ and $l \le x \le h$ | $f$, $\nabla f$ |
| `l_bfgs_qc_projected` | $\min f(x)$ subject to $\tfrac12 x^{\mathsf T} B_j x + \langle a_j, x\rangle = c_j^\star$ and $l \le x \le h$ | $f$, $\nabla f$, products $B_j u$ |
| `tr_newton_bounded` | $\min f(x)$ subject to $l \le x \le h$ | $f$, $\nabla f$, Hessian products |
| `constrained_conjugate_gradients` | quadratic $f$, $x \ge l$, optionally $\langle a, x\rangle = V^\star$ | $\nabla f$, Hessian products |
| `constrained_conjugate_gradients_with_restart` | quadratic $f$, $x \ge l$, optionally a prescribed mean of $x$ | $\nabla f$, Hessian products |

All return a SciPy-style `OptimizeResult` with at least `x`, `success`, `nit`
and `message`.

## MPI conventions

Arrays are *distributed*, scalars are *global*:

| Quantity | Lives where |
|----------|-------------|
| iterate `x`, initial guess `x0`, gradient | local slice on each process |
| bounds `bounds_lo`, `bounds_hi`, `bounds`, `zero_mask` | local, sliced like `x` |
| `LinearConstraint.a`, `QuadraticConstraint.a`, argument and result of its `hessp` | local |
| energy $f(x)$ | **global** (reduced) |
| `LinearConstraint.target` $V^\star$, `QuadraticConstraint.target` $c^\star$ | **global** |
| tolerances, Lagrange multiplier | **global** |
| argument of `callback(x)` | local slice of the current iterate |

User-supplied functions have to follow these rules:

- **The objective must return a globally reduced energy**, the same value on
  every process. Compute the local contribution and reduce it with
  [`Reduction`](mpi.md#reductions). A process-local energy is the most common
  mistake: the processes then disagree in the line-search tests and the
  iterates diverge silently.
- The gradient is local.
- `hessp(x, d)` returns the local slice of the Hessian-vector product.
- `callback(x)` receives the local slice; gather it yourself if you need the
  global state.

```python
import numpy as np
from NuMPI import MPI
from NuMPI.Tools import Reduction
from NuMPI.Optimization import l_bfgs_projected, LinearConstraint

comm = MPI.COMM_WORLD
pnp = Reduction(comm)

n_local = 100
y_local = np.random.default_rng(comm.rank).random(n_local)
a_local = np.ones(n_local)

# Sum of x over all processes is 10 * comm.size
constraint = LinearConstraint(a_local, 10.0 * comm.size, pnp=pnp)

def fun(x):
    energy = pnp.sum(0.5 * np.sum((x - y_local) ** 2)).item()  # reduced
    return energy, x - y_local                                 # local gradient

res = l_bfgs_projected(fun, np.full(n_local, 0.1), constraint, jac=True,
                       bounds_lo=0.0, bounds_hi=1.0, comm=comm, gtol=1e-8)
# res.x: local slice; res.fun, res.multiplier, res.max_grad: global
```

Pass either a communicator (`comm`) or a reduction wrapper (`pnp`) to the
L-BFGS and trust-region solvers, not both. The conjugate-gradient solvers take
`communicator`. `MPI_Quadratic` in
[`test/Optimization/MPIMinimizationProblems.py`](https://github.com/muSpectre/NuMPI/blob/main/test/Optimization/MPIMinimizationProblems.py)
is a reference implementation of a distributed objective.

## Objective and gradient

The L-BFGS and trust-region solvers accept the objective in SciPy's forms:

- `jac=True` (or `None` for `l_bfgs_bounded`, `l_bfgs_projected` and
  `tr_newton_bounded`): `fun(x, *args)` returns `(energy, gradient)`;
- `jac` callable: `fun(x, *args)` returns the energy and `jac(x, *args)` the
  gradient.

The conjugate-gradient solvers do not use the energy; with `jac=True` the energy
returned by `fun` can be a dummy value, and with `jac=False` `fun` returns only
the gradient.

## `l_bfgs`

```python
l_bfgs(fun, x, args=(), jac=None, x_old=None, maxcor=10, gtol=1e-5,
       ftol=0.0, xtol=0.0, maxiter=15000, maxls=20,
       linesearch_options=dict(c1=1e-3, c2=0.9), disp=None, pnp=None,
       comm=None, store_iterates=None, callback=None)
```

Unconstrained limited-memory BFGS with a strong-Wolfe line search. It converges
when the infinity norm of the gradient drops below `gtol`, or when the relative
change of the energy drops below `ftol`. `maxcor` is the number of correction
pairs kept, `maxls` the maximum number of line-search steps per iteration. With
`store_iterates`, the result contains the iterates. The result additionally
contains `fun`, `jac`, `max_grad` and `norm_grad`.

## `l_bfgs_bounded`

```python
l_bfgs_bounded(fun, x0, args=(), jac=None, bounds_lo=None, bounds_hi=None,
               zero_mask=None, gtol=1e-5, ftol=0.0, xtol=0.0, maxiter=500,
               maxcor=10, c1=1e-4, max_halvings=40, comm=None, pnp=None,
               callback=None, disp=False)
```

L-BFGS with box bounds, using the two-loop recursion and a projected Armijo
backtracking line search. `bounds_lo` and `bounds_hi` are scalars or local
arrays; `None` means unbounded. `zero_mask` pins entries to $x = 0$. It
converges when the infinity norm of the gradient on the free (not
bound-active) entries drops below `gtol`. If `callback(x)` returns `True`, the
optimization stops and is reported as successful, which allows custom stopping
criteria.

## `l_bfgs_projected`

```python
l_bfgs_projected(fun, x0, linear_constraint, args=(), jac=None,
                 bounds_lo=None, bounds_hi=None, zero_mask=None, gtol=1e-5,
                 ftol=0.0, xtol=0.0, maxiter=500, maxcor=10, c1=1e-4,
                 max_halvings=40, comm=None, pnp=None, callback=None,
                 disp=False)
```

L-BFGS on the set
$F = \{x : \langle a, x\rangle = V^\star,\ l \le x \le h,\ x_{\text{zero\_mask}} = 0\}$,
with the constraint given as a [`LinearConstraint`](#linearconstraint). Search
directions are projected onto the tangent of the constraint and the iterates
back onto $F$. It converges when the infinity norm of the projected gradient
drops below `gtol`, or when the largest change of an entry of $x$ drops below
`xtol`. The result contains the Lagrange multiplier of the constraint,
`multiplier`.

## `l_bfgs_qc_projected`

```python
l_bfgs_qc_projected(fun, x0, quadratic_constraint, args=(), jac=None,
                    bounds_lo=None, bounds_hi=None, zero_mask=None, gtol=1e-5,
                    ftol=0.0, xtol=0.0, maxiter=500, maxcor=10, c1=1e-4,
                    max_halvings=40, max_restore_rounds=10, comm=None,
                    pnp=None, callback=None, disp=False)
```

L-BFGS on the set

$$
F = \bigl\{x : c_j(x) = c_j^\star \ (j = 1, \dots, m),\ l \le x \le h,\
x_{\text{zero\_mask}} = 0\bigr\},
\qquad
c_j(x) = \tfrac12 x^{\mathsf T} B_j x + \langle a_j, x\rangle ,
$$

with the constraints given as a [`QuadraticConstraint`](#quadraticconstraint).
The matrices $B_j$ are constant and symmetric and enter only through their
products with vectors. The bilinear coupling $\sum_i g_i h_i = V$ of two fields
is the case $m = 1$, $x = (g, h)$, $a = 0$ and $B(u_g, u_h) = (u_h, u_g)$:

```python
def hessp(u):                      # u = (u_g, u_h), local slices
    u_g, u_h = np.split(u, 2)
    return np.concatenate([u_h, u_g])

constraint = QuadraticConstraint(np.zeros(2 * n_local), hessp, V, pnp=pnp)
res = l_bfgs_qc_projected(fun, np.concatenate([g0, h0]), constraint,
                          bounds_lo=0.0, comm=comm)
```

The feasible set is curved, so unlike in `l_bfgs_projected` the normals of the
constraints depend on the iterate: column $j$ of $N(x)$ is
$\nabla c_j(x) = B_j x + a_j$. One iteration at the feasible point $x$ with
gradient $g = \nabla f(x)$ does the following.

1. **Multipliers and reduced gradient.** With $\mathcal F$ the entries strictly
   inside their bounds and outside `zero_mask`, the multipliers are the
   least-squares fit of the gradient by the normals on $\mathcal F$,
   $\lambda = (N_{\mathcal F}^{\mathsf T} N_{\mathcal F})^{-1} N_{\mathcal F}^{\mathsf T} g_{\mathcal F}$,
   and the reduced gradient is $G = g - N \lambda$, the gradient of the
   Lagrangian $f - \sum_j \lambda_j c_j$. An entry is *held* if it is in
   `zero_mask`, or on its lower bound with $G_i \ge 0$, or on its upper bound
   with $G_i \le 0$. $G^{\mathrm p}$ is $G$ with the held entries set to zero,
   and the solver converges when $\lVert G^{\mathrm p}\rVert_\infty <$ `gtol`.
2. **Search direction.** The L-BFGS two-loop recursion is applied to
   $G^{\mathrm p}$. The result is set to zero on the held entries and projected
   onto the tangent space of the constraints on the remaining ones,
   $d \leftarrow d - N \mu$ with $\mu$ the least-squares fit of $d$ by the
   normals on the entries that are not held. If $d$ is not a descent direction,
   the curvature pairs are dropped and $d$ is the same projection of
   $-G^{\mathrm p}$.
3. **Line search.** The step $\alpha$ starts from 1 and is halved until the
   restored trial point $x^+ = R(x + \alpha d)$ satisfies the Armijo condition
   $f(x^+) \le f(x) + c_1 \alpha \langle G^{\mathrm p}, d\rangle$. $R$ is the
   restoration [`QuadraticConstraint.project`](#quadraticconstraint); a trial
   point that cannot be restored is rejected like one that does not decrease
   $f$ enough. If no step is accepted, the search is repeated once along the
   projected steepest descent, starting from
   $\alpha = 1 / \lVert G^{\mathrm p}\rVert_2$.
4. **Curvature pair.** $s = x^+ - x$ and $y = G^+ - P(x^+) G$, set to zero on
   the entries held at $x^+$. $P(x^+)$ removes the components along the
   normals at $x^+$, with the same fit as in step 1. Because the $B_j$ are
   constant, $y = P(x^+) \bigl[\nabla f(x^+) - \nabla f(x) - \sum_j \lambda_j B_j s\bigr]$:
   the pair measures the Hessian of the Lagrangian, including the curvature of
   the constraints, in the tangent space at $x^+$. That Hessian can be
   indefinite, and a pair is stored only if $\langle s, y\rangle$ is
   sufficiently positive.

`x0` is restored onto $F$ before the first iteration; if that fails, the solver
raises `RuntimeError`. The solver also converges when the largest change of an
entry of $x$ drops below `xtol`, or the relative change of $f$ below `ftol`.
The result contains the multipliers $\lambda$ (`multiplier`, one per
constraint), `max_grad` $= \lVert G^{\mathrm p}\rVert_\infty$ and
`constr_violation` $= \max_j \lvert c_j(x) - c_j^\star\rvert$.

Things to know:

- **Dependent normals.** The fits in steps 1 and 2 need the normals to be
  linearly independent on the entries they use. If the condition number of
  $N^{\mathsf T} N$ exceeds $10^{12}$ the solver stops with `success=False` and
  a message that the Gram matrix of the normals is singular. This happens where
  the feasible set is not smooth (for $\sum_i g_i h_i = V$ only at $g = h = 0$,
  hence only for $V = 0$), where the constraints are redundant, and where fewer
  entries than constraints are left inside the bounds.
- **Scaling.** Tangent projection and restoration use the Euclidean inner
  product of $x$. If the unknowns have different units or magnitudes (two
  fields $g$ and $h$, for example), scale them to comparable magnitude; the
  solver does not do this, and its convergence degrades quickly otherwise.
- **Local minima.** The feasible set is not convex. The solver finds a point
  that satisfies the first-order optimality conditions, and which one depends
  on `x0`.
- **Cost.** Besides one evaluation of `fun`, every trial point of the line
  search takes $m + 1$ calls of `hessp` per restoration round, and every
  iteration two more for the normals.

## `tr_newton_bounded`

```python
tr_newton_bounded(fun, x0, hessp, args=(), jac=None, bounds_lo=None,
                  bounds_hi=None, zero_mask=None, gtol=1e-5, maxiter=200,
                  delta0=0.05, delta_max=0.5, eta=0.1, inner_tol=None,
                  inner_maxiter=None, hessp_tol=False, fun_error=None,
                  request_accuracy=None, eta_f=0.25, max_accuracy_retries=5,
                  comm=None, pnp=None, callback=None, disp=False)
```

Bound-constrained trust-region Newton method with a truncated
(Steihaug–Toint) conjugate-gradient solver on the free entries. It needs only
products of a model Hessian $B$ with a vector, `hessp(x, v, *args)`; $B$ must
be symmetric and bounded but may be approximate. Convergence is measured as
for `l_bfgs_bounded`.

- `delta0` and `delta_max` are the initial and maximum trust-region radius per
  degree of freedom; the Euclidean radius is `delta * sqrt(N)` for $N$ global
  unknowns.
- A step is accepted if the ratio of actual to predicted decrease exceeds `eta`.
- `inner_tol` is the relative residual tolerance of the inner CG; by default it
  follows the forcing sequence $\min(0.5, \sqrt{\|g\| / \|g_0\|})$, relative to
  the projected gradient at the first iterate so that it does not depend on the
  units of the objective.
- With `hessp_tol=True`, `hessp` is called with the keyword `tol`, the inner
  CG's tolerance, so that a Hessian product that is itself computed inexactly
  need only be as accurate as the subproblem uses.
- For objectives that are evaluated inexactly (e.g. by truncated inner
  solves), `fun_error()` returns a bound on the error of the last evaluation and
  `request_accuracy(target)` asks the caller to evaluate more accurately. The
  solver keeps the error below `eta_f` times the predicted decrease.

The result additionally contains `nb_hessp` (number of Hessian products),
`delta_history` and `rho_history`.

### Arrays on the GPU

`tr_newton_bounded` works on arrays of any namespace that follows the
[Python array API standard](https://data-apis.org/array-api/), e.g. CuPy arrays
on a GPU (`pip install NuMPI[gpu]`). The iteration runs in the namespace and on
the device of `x0`: the iterate, the gradient, the bounds (array bounds are moved
there once), the free mask and the inner CG's vectors stay there, `fun`,
`hessp` and `callback` receive such arrays, and the result's `x` and `jac` are
such arrays. Only the reduced scalars -- dot products and maxima, single
numbers -- come to the host. `Reduction` reduces such arrays in their own
namespace too, but only completely (no `axis`), returning the result on the
host.

## `constrained_conjugate_gradients`

```python
constrained_conjugate_gradients(fun, hessp, x0, args=(), jac=True,
                                mean_val=None, linear_constraint=None,
                                gtol=1e-8, maxiter=3000, callback=None,
                                communicator=None, bounds=None)
```

Bound-constrained conjugate gradients for quadratic problems after Bugnicourt
et al., e.g. for contact mechanics. $x$ is constrained to $x \ge$ `bounds`
(zero if `None`; use `-inf` for unbounded entries). The conjugate direction is
not restarted when the active set changes. A linear equality is added with
`linear_constraint`, or, for uniform weights, with the shortcut `mean_val`
(prescribed mean of $x$; needs finite bounds everywhere). `hessp` may take
`(d)`, `(x, d)`, `(d, *args)` or `(x, d, *args)`. It converges when the
infinity norm of the projected gradient drops below `gtol`.

## `constrained_conjugate_gradients_with_restart`

```python
constrained_conjugate_gradients_with_restart(fun, hessp, x0, args=(),
                                             jac=True, mean_val=None,
                                             gtol=1e-8, maxiter=5000,
                                             callback=None, communicator=None,
                                             bounds=None, residual_plot=False)
```

Bound-constrained conjugate gradients after Polonsky & Keer, which restarts the
conjugate direction with steepest descent whenever the active set changes.
`mean_val` prescribes the global mean of $x$. The arguments are otherwise those
of `constrained_conjugate_gradients`. `residual_plot=True` plots the residual
history on rank 0 at convergence (needs matplotlib).

## `LinearConstraint`

```python
LinearConstraint(a, target, pnp=None)
```

The linear equality $\langle a, x\rangle = V^\star$ with non-negative weights
$a$ (the local slice) and the global right-hand side `target`. Pass
`pnp=Reduction(comm)` in parallel runs.

| Method | Result |
|--------|--------|
| `multiplier(grad, mask=None)` | Lagrange multiplier $\lambda = \langle a, g\rangle / \langle a, a\rangle$, optionally restricted to `mask` |
| `tangent(grad, mask=None)` | `grad - multiplier(grad, mask) * a`, the component of `grad` tangent to the constraint |
| `project(y, lo=None, hi=None, zero_mask=None, tol=1e-14, max_iter=60)` | Euclidean projection of `y` onto $F$ |
| `size` | length of the local slice of $a$ |

The projection has the closed form
$x_i = \operatorname{clip}(y_i - \mu a_i, l_i, h_i)$, where the multiplier
$\mu$ is the root of $\sum_i a_i \operatorname{clip}(y_i - \mu a_i, l_i, h_i) = V^\star$.
For $a \ge 0$ this sum is monotonic in $\mu$, and bisection finds $\mu$ in a
bounded number of reductions.

## `QuadraticConstraint`

```python
QuadraticConstraint(a, hessp, target, pnp=None)
```

The $m$ quadratic equalities
$c_j(x) = \tfrac12 x^{\mathsf T} B_j x + \langle a_j, x\rangle = c_j^\star$.
`a` is the local slice of the linear terms, an `(m, n)` array (or a vector for
$m = 1$). `hessp(u)` returns the local `(m, n)` array of the products
$B_j u$ for a local vector `u`; it may exchange ghost values, and every process
calls it the same number of times. `hessp=None` means $B_j = 0$. `target` holds
the global right-hand sides. Pass `pnp=Reduction(comm)` in parallel runs.

| Method | Result |
|--------|--------|
| `value(x)` | the values $c_j(x)$ |
| `normals(x)` | local `(n, m)` array $N(x)$ with columns $B_j x + a_j$ |
| `multiplier(grad, x, mask=None)` | least-squares multipliers $\lambda = (N^{\mathsf T} N)^{-1} N^{\mathsf T} g$ at `x`, with all sums restricted to `mask` |
| `tangent(grad, x, mask=None, return_multiplier=False)` | `grad` $- N \lambda$ |
| `project(y, lo=None, hi=None, zero_mask=None, max_rounds=10, max_newton=30)` | restoration of `y` onto the feasible set, or `None` |
| `check_hessp(rtol=1e-8, rng=None)` | raises `ValueError` unless `hessp` is linear and symmetric |
| `m` | number of constraints |

`multiplier` and `tangent` raise `numpy.linalg.LinAlgError` if the condition
number of $N^{\mathsf T} N$ on `mask` exceeds $10^{12}$.

**The constraint is quadratic by construction.** The class never evaluates the
constraint itself: value, normals and restoration are all computed from `a` and
`hessp`, and they describe the same function exactly if `hessp` is a constant
symmetric linear map. `check_hessp` tests this with two random vectors $u$, $v$:
linearity, $B_j (u - 2v) = B_j u - 2 B_j v$, and symmetry,
$\langle v, B_j u\rangle = \langle u, B_j v\rangle$, at the cost of three calls
of `hessp`. `l_bfgs_qc_projected` runs the test once before it starts. It
catches a product that depends on the point or is affine, and a quadratic form
represented by a non-symmetric matrix (for $\sum_i g_i h_i$, the product
$(u_h, 0)$ in place of $(u_h, u_g)$). It cannot tell whether the quadratic
function is the constraint you meant; compare `value(x)` with your own
evaluation of the constraint to test that. A constraint with terms beyond
second order is outside the scope of the class.

**Restoration.** `project` is *not* the Euclidean projection. It returns a
feasible point near `y` as follows.

1. Clip `y` to the bounds and set the entries in `zero_mask` to zero; call the
   result $\tilde y$. $\mathcal F$ are the entries of $\tilde y$ strictly inside
   their bounds and outside `zero_mask`.
2. Move along the normals at $\tilde y$, restricted to $\mathcal F$:
   $x = \tilde y + N_{\mathcal F}(\tilde y)\, t$ with $t \in \mathbb{R}^m$. Because the
   constraints are quadratic, their values along this move are the polynomials

    $$
    c_j(x) - c_j^\star = c_j(\tilde y) - c_j^\star
    + \bigl(N_{\mathcal F}^{\mathsf T} N_{\mathcal F}\, t\bigr)_j
    + \tfrac12\, t^{\mathsf T} \bigl(N_{\mathcal F}^{\mathsf T} B_j N_{\mathcal F}\bigr)\, t ,
    $$

    whose coefficients take $m + 1$ calls of `hessp` and one reduction. Newton's
    method from $t = 0$ finds the root of these $m$ equations in $m$ unknowns
    without touching the distributed arrays again.
3. If no entry of $x$ left its bounds, return $x$. Otherwise clip $x$, remove
   the entries that crossed a bound from $\mathcal F$, and repeat from step 2
   at the clipped point, at most `max_rounds` times.

`project` returns `None` if Newton's method does not converge (there may be no
real root along the normals, for example when `y` lies across the cone
$c(x) = 0$ from the feasible set), if the normals on $\mathcal F$ become
dependent, or if entries keep crossing bounds after `max_rounds`.

Without active bounds the restoration is a retraction that agrees with the
Euclidean projection to second order in the distance of `y` from the feasible
set (it moves along the normal space at $\tilde y$ instead of the one at $x$;
see Absil and Malick). For a sphere, $B = 1$ and $a = 0$, the two coincide.
With bounds they differ: an entry that is clipped in step 1 or 3 stays on its
bound, even where the correction along the normal would pull it back inside,
and step 3 continues from the clipped point, not from $\tilde y$. For the
linear constraint $x_1 + x_2 = 1$ on $[0, 1]^2$, the point $(1.2, 0.9)$ is
restored to $(1, 0)$, while its Euclidean projection, which
[`LinearConstraint.project`](#linearconstraint) returns, is $(0.65, 0.35)$. The
result is feasible in both cases; for the line search of `l_bfgs_qc_projected`
only that, and $R(x) = x$ to first order around feasible points, matters.

## References

- P.-A. Absil, J. Malick, *Projection-like retractions on matrix manifolds*,
  SIAM J. Optim. 22, 135 (2012),
  [doi:10.1137/100802529](https://doi.org/10.1137/100802529).

- J. Nocedal, *Updating quasi-Newton matrices with limited storage*,
  Math. Comp. 35, 773 (1980),
  [doi:10.1090/S0025-5718-1980-0572855-7](https://doi.org/10.1090/S0025-5718-1980-0572855-7).
- T. Steihaug, *The conjugate gradient method and trust regions in large scale
  optimization*, SIAM J. Numer. Anal. 20, 626 (1983),
  [doi:10.1137/0720042](https://doi.org/10.1137/0720042).
- R. Bugnicourt, P. Sainsot, D. Dureisseix, C. Gauthier, A. A. Lubrecht,
  *FFT-based methods for solving a rough adhesive contact: Description and
  convergence study*, Tribol. Lett. 66, 29 (2018),
  [doi:10.1007/s11249-017-0980-z](https://doi.org/10.1007/s11249-017-0980-z).
- I. A. Polonsky, L. M. Keer, *A numerical method for solving rough contact
  problems based on the multi-level multi-summation and conjugate gradient
  techniques*, Wear 231, 206 (1999),
  [doi:10.1016/S0043-1648(99)00113-1](https://doi.org/10.1016/S0043-1648(99)00113-1).
