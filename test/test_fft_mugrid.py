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

"""Cross-checks of NumpyFFTEngine against muGrid's FFTEngine"""

import numpy as np
import pytest

from NuMPI import MPI
from NuMPI.FFT import NumpyFFTEngine, make_fft_engine

muGrid = pytest.importorskip("muGrid")

GRIDS = [(8,), (7,), (8, 6), (7, 5), (4, 6, 5), (3, 5, 4)]


def _subdomain(array, locations, nb_grid_pts):
    """Slice the subdomain at `locations` from the trailing (spatial) axes."""
    spatial = tuple(slice(loc, loc + n) for loc, n in zip(locations, nb_grid_pts))
    return array[(Ellipsis,) + spatial]


@pytest.mark.parametrize("nb_grid_pts", GRIDS)
def test_grid_information_matches_mugrid(nb_grid_pts):
    numpy_engine = NumpyFFTEngine(nb_grid_pts)
    mugrid_engine = muGrid.FFTEngine(list(nb_grid_pts), communicator=MPI.COMM_SELF)
    for attr in ["spatial_dim", "nb_domain_grid_pts", "nb_subdomain_grid_pts", "subdomain_locations",
                 "nb_fourier_grid_pts", "nb_fourier_subdomain_grid_pts", "fourier_subdomain_locations"]:
        assert np.array_equal(getattr(numpy_engine, attr), getattr(mugrid_engine, attr)), attr
    np.testing.assert_allclose(numpy_engine.normalisation, mugrid_engine.normalisation)
    for attr in ["fftfreq", "ifftfreq", "coords", "icoords"]:
        np.testing.assert_allclose(getattr(numpy_engine, attr), np.asarray(getattr(mugrid_engine, attr)),
                                   err_msg=attr)


@pytest.mark.parametrize("nb_grid_pts", GRIDS)
@pytest.mark.parametrize("components", [(), (3,), (2, 2)])
def test_transforms_match_mugrid(nb_grid_pts, components):
    a = np.random.default_rng(0).random(components + nb_grid_pts)
    results = []
    for engine in [NumpyFFTEngine(nb_grid_pts), muGrid.FFTEngine(list(nb_grid_pts), communicator=MPI.COMM_SELF)]:
        real_field = engine.real_space_field("real", components=components)
        fourier_field = engine.fourier_space_field("fourier", components=components)
        assert real_field.p.shape == components + nb_grid_pts
        assert real_field.s.shape == components + (1,) + nb_grid_pts
        real_field.p[...] = a
        engine.fft(real_field, fourier_field)
        fourier = fourier_field.p.copy()
        engine.ifft(fourier_field, real_field)
        results.append((fourier, real_field.p * engine.normalisation))
    (numpy_fourier, numpy_real), (mugrid_fourier, mugrid_real) = results
    np.testing.assert_allclose(numpy_fourier, mugrid_fourier, atol=1e-12)
    np.testing.assert_allclose(numpy_real, mugrid_real, atol=1e-12)
    np.testing.assert_allclose(numpy_real, a)


def test_make_fft_engine_mugrid_backend():
    engine = make_fft_engine((8, 6), backend="muGrid")
    assert isinstance(engine, muGrid.FFTEngine)
    # No communicator means serial, also under an MPI launcher
    assert engine.communicator.size == 1


@pytest.mark.parametrize("nb_grid_pts", [(16, 12), (15, 11), (8, 9, 10)])
def test_parallel_transform_matches_numpy(comm, nb_grid_pts):
    if comm.Get_size() > 1 and not muGrid.has_mpi:
        pytest.skip("muGrid was compiled without MPI support")
    engine = make_fft_engine(nb_grid_pts, communicator=comm)
    if comm.Get_size() == 1:
        assert isinstance(engine, NumpyFFTEngine)
    else:
        assert isinstance(engine, muGrid.FFTEngine)
    # Every rank holds the same global data and transforms its subdomain
    a = np.random.default_rng(0).random(nb_grid_pts)
    real_field = engine.real_space_field("real")
    fourier_field = engine.fourier_space_field("fourier")
    real_field.p[...] = _subdomain(a, engine.subdomain_locations, engine.nb_subdomain_grid_pts)
    engine.fft(real_field, fourier_field)

    serial_engine = NumpyFFTEngine(nb_grid_pts)
    serial_real_field = serial_engine.real_space_field("real")
    serial_fourier_field = serial_engine.fourier_space_field("fourier")
    serial_real_field.p[...] = a
    serial_engine.fft(serial_real_field, serial_fourier_field)

    def fourier_subdomain(array):
        return _subdomain(array, engine.fourier_subdomain_locations, engine.nb_fourier_subdomain_grid_pts)

    np.testing.assert_allclose(fourier_field.p, fourier_subdomain(serial_fourier_field.p), atol=1e-12)
    np.testing.assert_allclose(np.asarray(engine.fftfreq), fourier_subdomain(serial_engine.fftfreq))
    np.testing.assert_allclose(engine.normalisation, serial_engine.normalisation)

    engine.ifft(fourier_field, real_field)
    np.testing.assert_allclose(real_field.p * engine.normalisation,
                               _subdomain(a, engine.subdomain_locations, engine.nb_subdomain_grid_pts))
