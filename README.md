NuMPI
=====

NuMPI is a collection of numerical tools for MPI-parallelized Python codes. NuMPI presently contains:

- An (incomplete) stub implementation of the [mpi4py](https://github.com/mpi4py/mpi4py) interface to the MPI libraries. This allows running serial versions of MPI parallel code  without having `mpi4py` (and hence a full MPI stack) installed.
- Parallel file IO in numpy's [.npy](https://docs.scipy.org/doc/numpy/reference/generated/numpy.save.html) format using MPI I/O.
- MPI-parallel L-BFGS optimizers:
  - `l_bfgs` — unconstrained, with a strong-Wolfe line search.
  - `l_bfgs_bounded` — box-constrained (`lo <= x <= hi`) with optional index pinning, two-loop recursion and projected Armijo backtracking.
  - `l_bfgs_projected` — a single linear equality `<a, x> = target` plus optional box bounds.
- An MPI-parallel bound constrained conjugate gradients algorithm.

Build status
------------

[![Tests](https://github.com/IMTEK-Simulation/NuMPI/actions/workflows/tests.yml/badge.svg?branch=master)](https://github.com/IMTEK-Simulation/NuMPI/actions/workflows/tests.yml) [![Flake8](https://github.com/IMTEK-Simulation/NuMPI/actions/workflows/flake8.yml/badge.svg?branch=master)](https://github.com/IMTEK-Simulation/NuMPI/actions/workflows/flake8.yml)

Installation
------------

```
python3 -m pip install NuMPI
```

Development Installation
------------------------

Clone the repository.

To use the code, install the current package as editable:

```
pip install -e .[test]
```

Testing
-------

You have to do a development installation to be able to run the tests.

We use [runtests](https://github.com/bccp/runtests). 

From the main installation directory:
```bash
python run-tests.py
```

If you want to use NuMPI without mpi4py, you can simply run the tests with pytest. 

```bash
pytest tests/
```

Testing on the cluster
----------------------
On NEMO for example

```bash
msub -q express -l walltime=15:00,nodes=1:ppn=20 NEMO_test_job.sh -m bea
```

MPI Conventions
---------------

All of NuMPI's parallel algorithms operate on *distributed* arrays: each MPI
rank holds a slice of the global data, and scalar quantities (energies, norms,
convergence tolerances, Lagrange multipliers) are *globally reduced* — the
same value on every rank. Understanding the split between *local* and *global*
is essential to using the optimizers correctly; this section spells it out.

### Distributed vs. global

| Quantity                                      | Lives where                     |
|-----------------------------------------------|---------------------------------|
| Iterate `x`, gradient `grad`, initial guess `x0` | *local* — each rank's own slice |
| Bounds `bounds_lo`, `bounds_hi`, `zero_mask`  | *local* — sliced to match `x`   |
| `LinearConstraint.a` (weight vector)          | *local*                         |
| Scalar energy `f(x)`                          | **global** (reduced)            |
| `LinearConstraint.target` (right-hand side)   | **global** (same on every rank) |
| Lagrange multiplier, convergence tolerance, `gtol`, `ftol` | **global**            |
| `callback(x)` argument                        | *local* slice of current iterate |

### User-supplied callbacks

The solvers call back into user code in a few places; each has a specific
contract.

* **Objective `fun(x) -> (energy, gradient)`** (when `jac=True`) or separate
  `fun(x) -> energy` and `jac(x) -> gradient`:

  - `energy` **must be a globally reduced scalar.** All ranks must return the
    same number. The standard way to do this is to compute a local quantity
    and reduce it with `pnp.sum(...).item()` (or equivalent), where `pnp` is
    the `Reduction(comm)` wrapper. Returning a *local* energy is the single
    most common MPI mistake: ranks will silently disagree in line-search
    acceptance tests and the optimisation will diverge or hang.
  - `gradient` is *local* — only the current rank's slice.

* **`callback(x)`** receives the current *local* iterate. If the caller needs
  the global state (for plotting or logging from rank 0), they must gather
  explicitly.

* **`hessp(x, d)`** (CG) returns a *local* Hessian-vector product.

### Building distributed inputs

Use `NuMPI.Tools.Reduction(comm)` to obtain a `pnp` object whose `sum`, `max`,
`min`, `mean`, `dot` methods perform `MPI_Allreduce` across the communicator.
When `mpi4py` is not installed, `NuMPI.MPIStub` provides the same interface
with a single "rank", so the same code runs serially too.

A typical setup with a communicator-provided subdomain looks like:

```python
from NuMPI.Tools import Reduction
from NuMPI.Optimization import LinearConstraint, l_bfgs_projected

pnp = Reduction(comm)

# a_local: this rank's slice of the global weight vector, shape matching x
# target: global scalar, same on every rank
lc = LinearConstraint(a_local, target, pnp=pnp)

def fun(x):                      # x is the local slice
    # compute local integrand, then REDUCE for the scalar return
    local_energy = 0.5 * np.sum((x - y_local) ** 2)
    return pnp.sum(local_energy).item(), (x - y_local)   # gradient stays local

res = l_bfgs_projected(fun, x0_local, lc, jac=True,
                       bounds_lo=0.0, bounds_hi=1.0,
                       comm=comm, gtol=1e-5)
```

The returned `res.x` is the local slice of the solution; `res.fun`,
`res.multiplier`, and `res.max_grad` are globally reduced scalars.

See `NuMPI/Optimization/__init__.py` for optimizer-specific notes and
`test/Optimization/MPIMinimizationProblems.py::MPI_Quadratic` for a
reference implementation of a distributed objective.

Development & Funding
---------------------

Development of this project is funded by the [European Research Council](https://erc.europa.eu) within [Starting Grant 757343](https://cordis.europa.eu/project/id/757343) and by the [Deutsche Forschungsgemeinschaft](https://www.dfg.de/en) within project [EXC 2193](https://gepris.dfg.de/gepris/projekt/390951807).
