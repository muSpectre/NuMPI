# FFT engines

```python
from NuMPI.FFT import make_fft_engine, NumpyFFTEngine
```

NuMPI's FFT engines compute real-to-complex Fourier transforms of 1D, 2D and 3D
grids. They have the interface and conventions of
[muGrid](https://github.com/muSpectre/muGrid)'s `FFTEngine` (muGrid >= 1.6.0),
so code written for muGrid's FFTs runs serially without muGrid installed.

## Choosing an engine

```python
make_fft_engine(nb_domain_grid_pts, communicator=None, backend="auto", **kwargs)
```

| `backend` | Engine |
|-----------|--------|
| `"auto"` (default) | `NumpyFFTEngine` for a single process, `muGrid.FFTEngine` for several |
| `"numpy"` | `NumpyFFTEngine`; the communicator must have a single process |
| `"muGrid"` | `muGrid.FFTEngine`, also for a single process |

Without a communicator, the engine is serial. Parallel transforms need muGrid
compiled with MPI support (`pip install NuMPI[fft]`, see
[Installation](installation.md)); without muGrid, `make_fft_engine` raises
`ImportError` for communicators with several processes. Further keyword
arguments are passed to the engine's constructor, e.g. `nb_ghosts_left` or
`device` for muGrid.

## Conventions

| | |
|-|-|
| Forward transform | `fft(real_field, fourier_field)` |
| Inverse transform | `ifft(fourier_field, real_field)`, **unnormalized** |
| Normalization | multiply the result of `ifft` by `normalisation` $= 1/N$ |
| Fourier layout | half-complex along the **first** axis: `np.fft.rfftn(a, axes=(dim - 1, ..., 0))` |
| Fourier grid | `nb_fourier_grid_pts` $= (n_0/2 + 1, n_1, \dots)$ |
| `fftfreq` | frequencies in units of the inverse grid spacing, shape `(dim, *nb_fourier_grid_pts)` |
| `ifftfreq` | integer wavevector indices, same shape |
| `coords`, `icoords` | coordinates of the real-space grid points, as fractions of the domain and as integers |

The wavevectors of a grid with spacing `pixel_size` (one value per dimension)
are

```python
q = 2 * np.pi * (engine.fftfreq.T / pixel_size).T
```

## Fields

Data is transformed between *fields* that belong to the engine:

```python
real_field = engine.real_space_field("heights")
fourier_field = engine.fourier_space_field("heights_q")
```

- A field is created at the first request and returned again for the same name.
  `register_real_space_field` and `register_fourier_space_field` create a field
  and raise if the name exists.
- `components` gives the shape of the values per grid point, e.g.
  `components=(3,)` for a vector field; all components are transformed.
- Real-space fields are `float64` or `float32` (`dtype`), Fourier-space fields
  `complex128` or `complex64`.
- `field.p` is the array of shape `(*components, *grid)`, `field.s`
  additionally has an axis for sub-points of length one.

## Example

```python
import numpy as np
from NuMPI import MPI
from NuMPI.FFT import make_fft_engine

comm = MPI.COMM_WORLD
nb_grid_pts = (128, 128)
pixel_size = np.array([0.1, 0.1])

engine = make_fft_engine(nb_grid_pts, communicator=comm)

# Every process fills its subdomain of the real-space grid
x, y = (engine.icoords.T * pixel_size).T
real_field = engine.real_space_field("heights")
real_field.p[...] = np.sin(2 * np.pi * x / (nb_grid_pts[0] * pixel_size[0]))

fourier_field = engine.fourier_space_field("heights_q")
engine.fft(real_field, fourier_field)

# Derivative along x in Fourier space
q = 2 * np.pi * (engine.fftfreq.T / pixel_size).T
fourier_field.p[...] *= 1j * q[0]
engine.ifft(fourier_field, real_field)
slope_x = real_field.p * engine.normalisation
```

The same code runs serially and under `mpirun`.

## Domain decomposition

For several processes, muGrid decomposes the grid; NuMPI does not choose the
decomposition. Every process holds the subdomain of shape
`nb_subdomain_grid_pts` at `subdomain_locations` in real space, and of shape
`nb_fourier_subdomain_grid_pts` at `fourier_subdomain_locations` in Fourier
space. Distribute your data according to these attributes, e.g. when reading
with [`load_npy`](io.md#reading):

```python
local = load_npy("heights.npy", subdomain_locations=engine.subdomain_locations,
                 nb_subdomain_grid_pts=engine.nb_subdomain_grid_pts, comm=comm)
```

muGrid splits 2D grids along their last axis; 3D grids are split along the last
axis (slabs) or the last two axes (pencils), see muGrid's documentation. For a
single process, the subdomain is the whole grid.
