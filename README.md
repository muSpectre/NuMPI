# NuMPI

[![Tests](https://github.com/muSpectre/NuMPI/actions/workflows/tests.yml/badge.svg?branch=main)](https://github.com/muSpectre/NuMPI/actions/workflows/tests.yml) [![Flake8](https://github.com/muSpectre/NuMPI/actions/workflows/flake8.yml/badge.svg?branch=main)](https://github.com/muSpectre/NuMPI/actions/workflows/flake8.yml) [![Docs](https://github.com/muSpectre/NuMPI/actions/workflows/docs.yml/badge.svg?branch=main)](https://muspectre.github.io/NuMPI/)

Numerical tools for MPI-parallelized Python codes. Code written with NuMPI runs
unchanged in parallel under `mpirun` and serially without any MPI installation.

- **MPI is optional**: `NuMPI.MPI` is mpi4py's `MPI` module if mpi4py is
  installed, and otherwise a stub with the same interface for a single process.
- **Reductions** (`sum`, `max`, `min`, `mean`, `dot`, `any`, `all`) over
  distributed arrays, for mpi4py communicators and for muGrid's `Communicator`.
- **Parallel I/O** of domain-decomposed arrays in numpy's `.npy` format with
  MPI I/O.
- **FFT engines** with the interface of [muGrid](https://github.com/muSpectre/muGrid)'s
  `FFTEngine`: a serial engine based on `numpy.fft`, and muGrid's MPI-parallel
  engine for domain-decomposed grids.
- **Parallel optimizers**: L-BFGS (unconstrained, bound-constrained, and with a
  linear equality), bound-constrained trust-region Newton-CG and constrained
  conjugate gradients, all on distributed arrays.
- **Testing utilities**: pytest fixtures for communicators of several sizes and
  assertions that agree across processes.

## Installation

```bash
pip install NuMPI
```

NuMPI requires Python 3.10 or later. The `mpi` extra installs mpi4py, the `fft`
extra muGrid for parallel FFTs:

```bash
pip install NuMPI[mpi,fft]
```

> **Note:** mpi4py and muGrid must be compiled against the MPI library of your
> system. Build them from source:
>
> ```bash
> CC=mpicc CXX=mpicxx pip install --no-binary mpi4py --no-binary muGrid NuMPI[mpi,fft]
> ```

## Quick start

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
`mpirun -n 4 python example.py`.

## Development

```bash
git clone https://github.com/muSpectre/NuMPI.git
cd NuMPI
pip install -e .[test]
python -m pytest                                  # serially, with the MPI stub
mpirun --oversubscribe -n 4 python -m pytest      # in parallel, needs mpi4py
```

## Documentation

Full documentation (installation, communicators and reductions, parallel I/O,
FFT engines, optimizers and their MPI conventions, testing utilities) is at
**https://muspectre.github.io/NuMPI/** and in the [`docs/`](docs/) folder.

## License

MIT — see [LICENSE.md](LICENSE.md).

## Funding

Development of this project is funded by the [European Research Council](https://erc.europa.eu) within [Starting Grant 757343](https://cordis.europa.eu/project/id/757343) and by the [Deutsche Forschungsgemeinschaft](https://www.dfg.de/en) within project [EXC 2193](https://gepris.dfg.de/gepris/projekt/390951807).
