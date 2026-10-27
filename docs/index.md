# NuMPI

NuMPI is a collection of numerical tools for MPI-parallelized Python codes.
Code written with NuMPI runs unchanged in parallel under `mpirun` and serially
without any MPI installation.

## Highlights

- **MPI is optional.** `NuMPI.MPI` is mpi4py's `MPI` module if mpi4py is
  installed, and otherwise a stub with the same interface for a single process.
- **Reductions on distributed arrays.** `Reduction(comm)` provides `sum`,
  `max`, `min`, `mean`, `dot`, `any` and `all` over all processes, for mpi4py
  communicators and for muGrid's `Communicator`.
- **Parallel I/O.** `save_npy` and `load_npy` write and read domain-decomposed
  arrays in numpy's `.npy` format with MPI I/O; every process handles its own
  subdomain of a single file.
- **FFT engines.** `make_fft_engine` returns a serial numpy-based engine for one
  process and muGrid's MPI-parallel engine for several, both with muGrid's
  interface.
- **Parallel optimizers.** L-BFGS (unconstrained, bound-constrained, and with a
  linear equality), a bound-constrained trust-region Newton-CG and constrained
  conjugate gradients, all operating on distributed arrays.
- **Testing utilities.** pytest fixtures for communicators of several sizes and
  assertions that agree across processes.

## A first example

```python
import numpy as np
from NuMPI import MPI
from NuMPI.IO import save_npy
from NuMPI.Tools import Reduction

comm = MPI.COMM_WORLD
pnp = Reduction(comm)

# Each process holds 10 rows of a global array with 10 * comm.size rows
local = np.full((10, 4), comm.rank, dtype=float)
print(pnp.sum(local))  # same value on every process

save_npy("data.npy", local, subdomain_locations=(10 * comm.rank, 0),
         nb_grid_pts=(10 * comm.size, 4), comm=comm)
```

Run it serially with `python example.py`, or in parallel with
`mpirun -n 4 python example.py`. The file `data.npy` can be read with
`numpy.load`.

## Where to go next

- [Installation](installation.md) — `pip install NuMPI`, and the optional
  `mpi` and `fft` extras.
- [Communicators & reductions](mpi.md) — the MPI stub, the communicators NuMPI
  accepts, and reductions over distributed arrays.
- [Parallel I/O](io.md) — reading and writing domain-decomposed `.npy` files.
- [FFT engines](fft.md) — Fourier transforms of serial and domain-decomposed
  grids.
- [Optimizers](optimization.md) — the parallel minimizers and the conventions
  for distributed objectives.
- [Testing utilities](testing.md) — pytest fixtures and parallel assertions.
- [Changelog](changelog.md)
