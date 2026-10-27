#
# Copyright 2026 Lars Pastewka
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
FFT engines with the interface of muGrid's ``FFTEngine``.

:class:`NumpyFFTEngine` is a serial engine built on :mod:`numpy.fft` that
mirrors the parts of ``muGrid.FFTEngine`` (muGrid >= 1.6.0) used for
real-to-complex transforms, so that code written against muGrid runs
without muGrid installed:

- The Fourier-space field is half-complex along the *first* spatial axis,
  i.e. ``fft`` computes ``np.fft.rfftn(a, axes=(dim - 1, ..., 0))``.
- ``ifft`` is unnormalized; multiply by ``normalisation`` (``1 / N``) to
  invert ``fft``.
- ``fftfreq`` has shape ``(dim, *nb_fourier_grid_pts)`` and contains the
  frequencies in units of the inverse grid spacing; ``ifftfreq`` contains
  the corresponding integer wavevector indices.
- Fields are obtained by name from ``real_space_field`` and
  ``fourier_space_field``; ``.p`` is the pixel view of shape
  ``(*components, *grid)``, ``.s`` additionally has a sub-point axis.
"""

import numpy as np

from . import MPI


def _comm_size(comm):
    return comm.Get_size() if hasattr(comm, "Get_size") else comm.size


class NumpyField:
    """
    Field of an :class:`NumpyFFTEngine`, with the array accessors of
    muGrid's ``Field``.
    """

    def __init__(self, name, components, nb_grid_pts, dtype):
        self._name = name
        self._components = tuple(components)
        self._data = np.zeros(self._components + tuple(nb_grid_pts), dtype=dtype)

    @property
    def name(self):
        """Name of the field."""
        return self._name

    @property
    def components(self):
        """Shape of the field components."""
        return self._components

    @property
    def dtype(self):
        """Data type of the field."""
        return self._data.dtype

    @property
    def p(self):
        """Pixel view, shape ``(*components, *grid)``."""
        return self._data

    @property
    def s(self):
        """Sub-point view, shape ``(*components, 1, *grid)``."""
        return self._data.reshape(self._components + (1,) + self._data.shape[len(self._components):])

    # Serial engines have no ghost layers, the views including ghosts are
    # the plain views
    pg = p
    sg = s


class NumpyFFTEngine:
    """
    Serial real-to-complex FFT engine based on :mod:`numpy.fft`.

    It has the interface (and conventions) of ``muGrid.FFTEngine``, see the
    module documentation, but runs only on a single process.

    Parameters
    ----------
    nb_domain_grid_pts : sequence of int
        Number of grid points in each spatial direction (1, 2 or 3).
    communicator : communicator, optional
        Communicator; must have a single process. (Default: MPI.COMM_SELF)
    """

    def __init__(self, nb_domain_grid_pts, communicator=None):
        self._nb_domain_grid_pts = tuple(int(n) for n in nb_domain_grid_pts)
        if len(self._nb_domain_grid_pts) not in (1, 2, 3):
            raise ValueError(
                "NumpyFFTEngine supports 1, 2 or 3 spatial dimensions, got grid {}.".format(
                    self._nb_domain_grid_pts))
        if communicator is None:
            communicator = MPI.COMM_SELF
        if _comm_size(communicator) != 1:
            raise ValueError(
                "NumpyFFTEngine is serial, but the communicator has {} processes.".format(
                    _comm_size(communicator)))
        self._communicator = communicator
        self._real_space_fields = {}
        self._fourier_space_fields = {}

    # -- Grid information ------------------------------------------------- #

    @property
    def communicator(self):
        """The communicator the engine was constructed with."""
        return self._communicator

    @property
    def spatial_dim(self):
        """Number of spatial dimensions."""
        return len(self._nb_domain_grid_pts)

    @property
    def nb_domain_grid_pts(self):
        """Number of grid points of the whole domain."""
        return self._nb_domain_grid_pts

    @property
    def nb_subdomain_grid_pts(self):
        """Number of grid points of the local subdomain (the whole domain)."""
        return self._nb_domain_grid_pts

    @property
    def subdomain_locations(self):
        """Location of the local subdomain within the domain."""
        return (0,) * self.spatial_dim

    @property
    def nb_fourier_grid_pts(self):
        """Number of grid points of the half-complex Fourier domain."""
        return (self._nb_domain_grid_pts[0] // 2 + 1,) + self._nb_domain_grid_pts[1:]

    @property
    def nb_fourier_subdomain_grid_pts(self):
        """Number of grid points of the local Fourier subdomain."""
        return self.nb_fourier_grid_pts

    @property
    def fourier_subdomain_locations(self):
        """Location of the local Fourier subdomain within the Fourier domain."""
        return (0,) * self.spatial_dim

    @property
    def normalisation(self):
        """Factor (``1 / N``) that makes ``ifft`` the inverse of ``fft``."""
        return 1 / np.prod(self._nb_domain_grid_pts)

    @property
    def ifftfreq(self):
        """Integer wavevector indices, shape ``(dim, *nb_fourier_grid_pts)``."""
        n0 = self._nb_domain_grid_pts[0]
        indices = [np.arange(n0 // 2 + 1)] + \
            [np.fft.fftfreq(n, 1 / n).astype(int) for n in self._nb_domain_grid_pts[1:]]
        return np.array(np.meshgrid(*indices, indexing="ij"))

    @property
    def fftfreq(self):
        """Frequencies in units of the inverse grid spacing, shape
        ``(dim, *nb_fourier_grid_pts)``."""
        n = np.array(self._nb_domain_grid_pts).reshape((-1,) + (1,) * self.spatial_dim)
        return self.ifftfreq / n

    @property
    def icoords(self):
        """Integer coordinates of the real-space grid points, shape
        ``(dim, *nb_subdomain_grid_pts)``."""
        return np.indices(self._nb_domain_grid_pts)

    @property
    def coords(self):
        """Coordinates of the real-space grid points in units of the domain
        size, shape ``(dim, *nb_subdomain_grid_pts)``."""
        n = np.array(self._nb_domain_grid_pts).reshape((-1,) + (1,) * self.spatial_dim)
        return self.icoords / n

    # -- Fields ------------------------------------------------------------ #

    @staticmethod
    def _get_field(fields, name, components, nb_grid_pts, dtype):
        components = tuple(components)
        try:
            field = fields[name]
        except KeyError:
            field = fields[name] = NumpyField(name, components, nb_grid_pts, dtype)
            return field
        if field.components != components:
            raise ValueError(
                "Field '{}' exists with components {}, cannot change them to {}.".format(
                    name, field.components, components))
        return field

    @staticmethod
    def _register_field(fields, name, components, nb_grid_pts, dtype):
        if name in fields:
            raise ValueError("A field of name '{}' is already registered.".format(name))
        field = fields[name] = NumpyField(name, components, nb_grid_pts, dtype)
        return field

    @staticmethod
    def _check_sub_pt(sub_pt):
        if sub_pt != "pixel":
            raise NotImplementedError("NumpyFFTEngine only supports fields on pixels.")

    @staticmethod
    def _real_dtype(dtype):
        dtype = np.dtype(dtype)
        if dtype not in (np.float64, np.float32):
            raise ValueError("Real-space fields support dtype float64 or float32, got {}.".format(dtype))
        return dtype

    @staticmethod
    def _complex_dtype(dtype):
        dtype = np.dtype(dtype)
        if dtype not in (np.complex128, np.complex64):
            raise ValueError("Fourier-space fields support dtype complex128 or complex64, got {}.".format(dtype))
        return dtype

    def real_space_field(self, name, components=(), dtype=np.float64, sub_pt="pixel"):
        """Get or create the real-space field ``name``."""
        self._check_sub_pt(sub_pt)
        return self._get_field(self._real_space_fields, name, components,
                               self.nb_subdomain_grid_pts, self._real_dtype(dtype))

    def fourier_space_field(self, name, components=(), dtype=np.complex128, sub_pt="pixel"):
        """Get or create the Fourier-space field ``name``."""
        self._check_sub_pt(sub_pt)
        return self._get_field(self._fourier_space_fields, name, components,
                               self.nb_fourier_subdomain_grid_pts, self._complex_dtype(dtype))

    def register_real_space_field(self, name, components=(), dtype=np.float64, sub_pt="pixel"):
        """Create the real-space field ``name``; it must not exist yet."""
        self._check_sub_pt(sub_pt)
        return self._register_field(self._real_space_fields, name, components,
                                    self.nb_subdomain_grid_pts, self._real_dtype(dtype))

    def register_fourier_space_field(self, name, components=(), dtype=np.complex128, sub_pt="pixel"):
        """Create the Fourier-space field ``name``; it must not exist yet."""
        self._check_sub_pt(sub_pt)
        return self._register_field(self._fourier_space_fields, name, components,
                                    self.nb_fourier_subdomain_grid_pts, self._complex_dtype(dtype))

    # -- Transforms -------------------------------------------------------- #

    def _check_fields(self, real_field, fourier_field):
        for field, nb_grid_pts, kind in ((real_field, self.nb_subdomain_grid_pts, "real-space"),
                                         (fourier_field, self.nb_fourier_subdomain_grid_pts, "Fourier-space")):
            if not isinstance(field, NumpyField):
                raise TypeError("Expected a {} field of a NumpyFFTEngine, got {}.".format(kind, type(field)))
            if field.p.shape[len(field.components):] != nb_grid_pts:
                raise ValueError("The {} field '{}' has grid {}, but the engine expects {}.".format(
                    kind, field.name, field.p.shape[len(field.components):], nb_grid_pts))
        if real_field.components != fourier_field.components:
            raise ValueError("Real-space field has components {}, but Fourier-space field has {}.".format(
                real_field.components, fourier_field.components))

    def _axes(self, field):
        # Spatial axes in reverse order: the real (half-complex) transform is
        # carried out along the last axis listed, i.e. the first spatial axis
        nb_components_dims = len(field.components)
        return tuple(range(nb_components_dims + self.spatial_dim - 1, nb_components_dims - 1, -1))

    def fft(self, real_field, fourier_field):
        """Forward transform of ``real_field`` into ``fourier_field``."""
        self._check_fields(real_field, fourier_field)
        fourier_field.p[...] = np.fft.rfftn(real_field.p, axes=self._axes(real_field))

    def ifft(self, fourier_field, real_field):
        """Unnormalized inverse transform of ``fourier_field`` into
        ``real_field``; multiply by ``normalisation`` to invert ``fft``."""
        self._check_fields(real_field, fourier_field)
        axes = self._axes(real_field)
        s = [self._nb_domain_grid_pts[a - len(real_field.components)] for a in axes]
        real_field.p[...] = np.fft.irfftn(fourier_field.p, s=s, axes=axes) * np.prod(self._nb_domain_grid_pts)
