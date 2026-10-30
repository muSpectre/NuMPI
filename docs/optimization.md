# Optimizers

```python
from NuMPI.Optimization import (
    l_bfgs, l_bfgs_bounded, l_bfgs_projected, tr_newton_bounded,
    constrained_conjugate_gradients, constrained_conjugate_gradients_with_restart,
    LinearConstraint,
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
| `LinearConstraint.a` | local |
| energy $f(x)$ | **global** (reduced) |
| `LinearConstraint.target` $V^\star$ | **global** |
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

## References

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
