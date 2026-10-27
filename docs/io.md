# Parallel I/O

```python
from NuMPI.IO import save_npy, load_npy, mpi_open
```

NuMPI reads and writes arrays in numpy's
[`.npy` format](https://numpy.org/doc/stable/reference/generated/numpy.lib.format.html)
with MPI I/O. The array is decomposed into subdomains; every process writes or
reads its own subdomain of a single file. The files are ordinary `.npy` files
that `numpy.load` and `numpy.save` read and write.

Without mpi4py, the same functions run serially on the
[MPI stub](mpi.md#the-mpi-stub).

## Writing

```python
save_npy(fn, data, subdomain_locations=None, nb_grid_pts=None,
         components_are_leading=True, comm=MPI.COMM_WORLD)
```

Every process passes its local subdomain `data`, the location of its first
element in the global array, `subdomain_locations`, and the shape of the global
array, `nb_grid_pts`. Without `subdomain_locations` and `nb_grid_pts`, `data` is
the whole array.

```python
import numpy as np
from NuMPI import MPI
from NuMPI.IO import save_npy

comm = MPI.COMM_WORLD
nx, ny = 8 * comm.size, 5

# Slab decomposition along the first axis
local = np.arange(8 * ny, dtype=float).reshape(8, ny) + 100 * comm.rank
save_npy("slabs.npy", local, subdomain_locations=(8 * comm.rank, 0),
         nb_grid_pts=(nx, ny), comm=comm)
```

- The subdomains of all processes must tile the global array: no gaps and no
  overlaps. `save_npy` checks this and raises `ValueError` otherwise (on every
  process).
- `data` must be contiguous. It is written in its memory order: a
  Fortran-contiguous array produces a file with `fortran_order: True`, without
  copying the data.
- A process can hold an empty subdomain, e.g. when there are more processes
  than rows.

## Reading

```python
load_npy(fn, subdomain_locations=None, nb_subdomain_grid_pts=None,
         components_are_leading=True, comm=MPI.COMM_WORLD)
```

Every process reads the subdomain of shape `nb_subdomain_grid_pts` at
`subdomain_locations`. Without them, every process reads the whole array.

```python
from NuMPI.IO import load_npy

nb_local_rows = nx // comm.size
local = load_npy("slabs.npy", subdomain_locations=(nb_local_rows * comm.rank, 0),
                 nb_subdomain_grid_pts=(nb_local_rows, ny), comm=comm)
```

On reading, the subdomains need not tile the array: processes may read
overlapping regions or only part of the file. A subdomain that extends beyond
the array raises `ValueError`.

To inspect a file before reading it, open it with `mpi_open`:

```python
file = mpi_open("slabs.npy", comm)   # NPYFile
file.array_shape    # (nx, ny)
file.dtype          # dtype('float64')
file.fortran_order  # False
local = file.read(subdomain_locations=(0, 0), nb_subdomain_grid_pts=(2, ny))
file.close()
```

`mpi_open` raises `MPIFileTypeError` if the file is not an `.npy` file. With
the MPI stub, `fn` can also be a Python file object opened in binary mode;
mpi4py's MPI I/O needs a file name.

## Components

Arrays can have components per grid point, e.g. a vector field of shape
`(3, nx, ny)`. Only the spatial dimensions are decomposed. The number of
spatial dimensions is the length of `subdomain_locations` (or `nb_grid_pts`,
`nb_subdomain_grid_pts`); the remaining dimensions are components. They lead
(`(*components, *grid)`, the default) or trail (`(*grid, *components)`,
`components_are_leading=False`):

```python
# Vector field with three components; global shape (3, nx, ny)
local_field = np.zeros((3, 8, ny))
save_npy("field.npy", local_field, subdomain_locations=(8 * comm.rank, 0),
         nb_grid_pts=(nx, ny), comm=comm)
```

## Data types

All numpy data types with a fixed size can be written and read, in either byte
order. The returned array has the data type of the file, as with `numpy.load`;
e.g. a big-endian file gives a `>f8` array. Data types the MPI library does not
provide, such as half precision with some MPI libraries, are transferred as raw
bytes.
