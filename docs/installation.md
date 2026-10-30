# Installation

## From PyPI

```bash
pip install NuMPI
```

NuMPI requires Python 3.10 or later and depends only on numpy, scipy,
primefac and array-api-compat. Without mpi4py, NuMPI uses its [MPI stub](mpi.md#the-mpi-stub) and
runs serially.

## Optional dependencies

| Extra | Installs | Needed for |
|-------|----------|------------|
| `mpi` | mpi4py | running in parallel under `mpirun` |
| `fft` | muGrid (>= 1.6.0) | [FFTs](fft.md) of domain-decomposed grids |
| `gpu` | CuPy | [optimization on the GPU](optimization.md#arrays-on-the-gpu) |
| `test` | pytest, pytest-cov, array-api-strict | running the test suite |

The `gpu` extra installs the generic `cupy` package, which compiles from
source. A prebuilt CuPy wheel for your platform (e.g. `cupy-cuda12x` or
`cupy-rocm-6-0`) installed instead works as well.

```bash
pip install NuMPI[mpi,fft]
```

!!! warning "Compile mpi4py and muGrid against your MPI library"
    The wheels of mpi4py and muGrid on PyPI may be built against a different
    MPI library than the one your `mpirun` belongs to, or without MPI at all.
    Build both from source against the MPI library of your system:

    ```bash
    CC=mpicc CXX=mpicxx pip install --no-binary mpi4py --no-binary muGrid NuMPI[mpi,fft]
    ```

    muGrid refuses communicators from an mpi4py linked against a different MPI
    library, and `muGrid.has_mpi` tells whether muGrid was compiled with MPI.

## Development installation

Clone the repository and install it in editable mode:

```bash
git clone https://github.com/muSpectre/NuMPI.git
cd NuMPI
pip install -e .[test]
```

## Running the tests

```bash
python -m pytest
```

This runs serially and exercises the MPI stub, so it works without mpi4py.
With mpi4py installed, run the suite in parallel with

```bash
mpirun --oversubscribe -n 4 python -m pytest
```

CI runs the suite serially without mpi4py, and with mpi4py on 1, 2, 4, 8 and 10
processes. Tests that need a communicator of a given size are skipped when
there are fewer processes (see [Testing utilities](testing.md)).

## Building this documentation

```bash
pip install -r docs/requirements.txt
mkdocs serve
```
