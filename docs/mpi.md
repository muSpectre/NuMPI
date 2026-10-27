# Communicators & reductions

## The MPI stub

```python
from NuMPI import MPI
```

`NuMPI.MPI` is mpi4py's `MPI` module if mpi4py can be imported. Otherwise it is
`NuMPI.MPIStub`, an implementation of the parts of mpi4py's interface that
NuMPI and the codes built on it use, for a single process. `NuMPI._has_mpi4py`
tells which one is in use.

The stub provides

- `COMM_WORLD` and `COMM_SELF`, both with `rank == 0` and `size == 1`
  (`Get_rank()`, `Get_size()`, `Barrier()`/`barrier()`);
- the collectives `bcast`, `gather`, `allgather`, `scatter`, `reduce`,
  `allreduce`, and the buffer-based `Reduce`, `Allreduce` and `Allgather`,
  which with a single process copy or return their input;
- `File` with `Open`, `Set_view`, `Read`, `Read_all`, `Write`, `Write_all` and
  `Close`, so that [MPI I/O](io.md) works serially, including on Python file
  objects;
- the datatypes (`_typedict`, `Create_contiguous`, `Create_subarray`, …),
  reduction operations (`SUM`, `MAX`, `MIN`, …) and file modes
  (`MODE_RDONLY`, `MODE_WRONLY`, `MODE_CREATE`, …) these need.

Code that only uses these runs serially without an MPI installation:

```python
from NuMPI import MPI

comm = MPI.COMM_WORLD
if comm.rank == 0:
    print(f"Running on {comm.size} process(es)")
```

## Communicators

NuMPI's functions take a communicator argument (`comm` or `communicator`).
Three kinds are accepted:

| Communicator | Requires |
|--------------|----------|
| mpi4py communicators, e.g. `MPI.COMM_WORLD` or the result of `Split` | mpi4py |
| the stub's `COMM_WORLD`/`COMM_SELF` | nothing |
| muGrid's `Communicator`, e.g. `muGrid.Communicator.world()` | muGrid, but not mpi4py |

Communicators other than mpi4py's are used through a small protocol, so any
object that implements it works:

- `allreduce(array, op)` with `op` one of `"sum"`, `"max"`, `"min"`, returning
  an array of the same shape and data type on every process;
- `compare(other)`, returning `"identical"`, `"congruent"`, `"similar"` or
  `"unequal"`;
- the attributes `size` and `rank`.

## Reductions

```python
from NuMPI.Tools import Reduction

pnp = Reduction(comm)
```

`Reduction(comm)` wraps a communicator into an object with numpy-like
reductions over a distributed array: each process passes its local data, and
every process receives the global result. `Reduction()` without a
communicator reduces over a single process.

| Method | Result |
|--------|--------|
| `pnp.sum(arr, axis=None)` | global sum |
| `pnp.max(arr, axis=None)` | global maximum |
| `pnp.min(arr, axis=None)` | global minimum |
| `pnp.mean(arr)` | global mean, $\sum_p \sum_i a_i / \sum_p n_p$ |
| `pnp.dot(a, b)` | global dot product of two distributed vectors |
| `pnp.any(arr)`, `pnp.all(arr)` | logical reductions |

Positional and keyword arguments (e.g. `axis`) are passed to the local numpy
operation before the global reduction. The mean is the mean over all elements
of all processes, so it is correct also when processes hold different numbers
of elements.

```python
import numpy as np
from NuMPI import MPI
from NuMPI.Tools import Reduction

comm = MPI.COMM_WORLD
pnp = Reduction(comm)

local = np.random.default_rng(comm.rank).random(100 + comm.rank)
rms = np.sqrt(pnp.mean(local ** 2))  # same value on every process
```

`NuMPI.Tools.Reduction.same_reduction_group(pnp_a, pnp_b)` tells whether two
reduction wrappers reduce over the same group of processes. It compares the
communicators, not the wrapper objects, so two `Reduction` instances of the
same communicator count as the same group.

## Distributed and global quantities

NuMPI's parallel algorithms operate on *distributed* arrays: every process
holds a slice of the global data, and scalar quantities (energies, norms,
tolerances, Lagrange multipliers) are *global*, i.e. reduced and the same on
every process. The [optimizers](optimization.md#mpi-conventions) rely on this
split, and user code that feeds them must respect it.

## Domain decompositions

`NuMPI.Tools.Subdivision.suggest_subdivisions(nb_dims, nb_procs)` suggests the
number of subdivisions along each of `nb_dims` dimensions for `nb_procs`
processes, from the prime factors of `nb_procs`:

```python
from NuMPI.Tools.Subdivision import suggest_subdivisions

suggest_subdivisions(2, 12)  # [2, 6]
suggest_subdivisions(3, 4)   # [2, 2, 1]
```

The [FFT engines](fft.md) determine their own decomposition; use the
engine's `subdomain_locations` and `nb_subdomain_grid_pts` when data is
transformed with it.
