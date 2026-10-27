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

import numpy as np
import pytest

from NuMPI.FFT import NumpyFFTEngine

GRIDS = [(8,), (7,), (8, 6), (7, 5), (4, 6, 5), (3, 5, 4)]


@pytest.mark.parametrize("nb_grid_pts", GRIDS)
def test_numpy_engine_grid_information(nb_grid_pts):
    engine = NumpyFFTEngine(nb_grid_pts)
    dim = len(nb_grid_pts)
    nb_fourier = (nb_grid_pts[0] // 2 + 1,) + nb_grid_pts[1:]
    assert engine.spatial_dim == dim
    assert engine.nb_domain_grid_pts == nb_grid_pts
    assert engine.nb_subdomain_grid_pts == nb_grid_pts
    assert engine.subdomain_locations == (0,) * dim
    assert engine.nb_fourier_grid_pts == nb_fourier
    assert engine.nb_fourier_subdomain_grid_pts == nb_fourier
    assert engine.fourier_subdomain_locations == (0,) * dim
    np.testing.assert_allclose(engine.normalisation, 1 / np.prod(nb_grid_pts))


@pytest.mark.parametrize("nb_grid_pts", GRIDS)
def test_numpy_engine_fftfreq(nb_grid_pts):
    engine = NumpyFFTEngine(nb_grid_pts)
    dim = len(nb_grid_pts)
    nb_fourier = (nb_grid_pts[0] // 2 + 1,) + nb_grid_pts[1:]
    freqs = [np.fft.rfftfreq(nb_grid_pts[0])] + [np.fft.fftfreq(n) for n in nb_grid_pts[1:]]
    fftfreq = engine.fftfreq
    ifftfreq = engine.ifftfreq
    assert fftfreq.shape == (dim,) + nb_fourier
    assert ifftfreq.shape == (dim,) + nb_fourier
    assert np.issubdtype(ifftfreq.dtype, np.integer)
    for d in range(dim):
        # Frequencies vary only along their own axis
        index = [0] * dim
        index[d] = slice(None)
        np.testing.assert_allclose(fftfreq[d][tuple(index)], freqs[d])
        np.testing.assert_array_equal(ifftfreq[d][tuple(index)], np.round(freqs[d] * nb_grid_pts[d]))
    np.testing.assert_allclose(fftfreq * np.reshape(nb_grid_pts, (-1,) + (1,) * dim), ifftfreq)


@pytest.mark.parametrize("nb_grid_pts", GRIDS)
def test_numpy_engine_coords(nb_grid_pts):
    engine = NumpyFFTEngine(nb_grid_pts)
    dim = len(nb_grid_pts)
    assert engine.icoords.shape == (dim,) + nb_grid_pts
    np.testing.assert_array_equal(engine.icoords, np.indices(nb_grid_pts))
    np.testing.assert_allclose(engine.coords * np.reshape(nb_grid_pts, (-1,) + (1,) * dim), engine.icoords)


@pytest.mark.parametrize("nb_grid_pts", GRIDS)
@pytest.mark.parametrize("components", [(), (3,), (2, 2)])
def test_numpy_engine_fft_roundtrip(nb_grid_pts, components):
    engine = NumpyFFTEngine(nb_grid_pts)
    dim = len(nb_grid_pts)
    real_field = engine.real_space_field("real", components=components)
    fourier_field = engine.fourier_space_field("fourier", components=components)
    assert real_field.p.shape == components + nb_grid_pts
    assert real_field.s.shape == components + (1,) + nb_grid_pts
    assert fourier_field.p.shape == components + engine.nb_fourier_grid_pts

    rng = np.random.default_rng(0)
    a = rng.random(components + nb_grid_pts)
    real_field.p[...] = a
    engine.fft(real_field, fourier_field)
    # Half-complex along the first spatial axis
    axes = tuple(range(len(components) + dim - 1, len(components) - 1, -1))
    np.testing.assert_allclose(fourier_field.p, np.fft.rfftn(a, axes=axes))

    real_field.p[...] = 0
    engine.ifft(fourier_field, real_field)
    # ifft is unnormalized
    np.testing.assert_allclose(real_field.p * engine.normalisation, a)


def test_numpy_engine_single_precision():
    engine = NumpyFFTEngine((8, 6))
    real_field = engine.real_space_field("real", dtype=np.float32)
    fourier_field = engine.fourier_space_field("fourier", dtype=np.complex64)
    assert real_field.p.dtype == np.float32
    assert fourier_field.p.dtype == np.complex64
    a = np.random.default_rng(1).random((8, 6)).astype(np.float32)
    real_field.p[...] = a
    engine.fft(real_field, fourier_field)
    engine.ifft(fourier_field, real_field)
    np.testing.assert_allclose(real_field.p * engine.normalisation, a, rtol=1e-5)
    with pytest.raises(ValueError):
        engine.real_space_field("complex", dtype=np.complex128)
    with pytest.raises(ValueError):
        engine.fourier_space_field("real", dtype=np.float32)


def test_numpy_engine_fields_are_cached_by_name():
    engine = NumpyFFTEngine((8, 6))
    field = engine.real_space_field("u", components=(3,))
    assert engine.real_space_field("u", components=(3,)) is field
    assert engine.real_space_field("v") is not field
    # Real-space and Fourier-space fields live in separate collections
    assert engine.fourier_space_field("u", components=(3,)) is not field
    with pytest.raises(ValueError):
        engine.real_space_field("u", components=(2,))
    with pytest.raises(ValueError):
        engine.register_real_space_field("u", components=(3,))
    registered = engine.register_fourier_space_field("w")
    assert engine.fourier_space_field("w") is registered


def test_numpy_engine_rejects_mismatching_fields():
    engine = NumpyFFTEngine((8, 6))
    other = NumpyFFTEngine((4, 4))
    with pytest.raises(ValueError):
        engine.fft(other.real_space_field("real"), engine.fourier_space_field("fourier"))
    with pytest.raises(ValueError):
        engine.fft(engine.real_space_field("vector", components=(2,)), engine.fourier_space_field("fourier"))
    with pytest.raises(TypeError):
        engine.fft(np.zeros((8, 6)), engine.fourier_space_field("fourier"))


def test_numpy_engine_rejects_unsupported_arguments():
    with pytest.raises(ValueError):
        NumpyFFTEngine((2, 2, 2, 2))
    engine = NumpyFFTEngine((8, 6))
    with pytest.raises(NotImplementedError):
        engine.real_space_field("quad", sub_pt="quad")


def test_numpy_engine_requires_serial_communicator(comm):
    if comm.Get_size() == 1:
        assert NumpyFFTEngine((8, 6), communicator=comm).communicator is comm
    else:
        with pytest.raises(ValueError):
            NumpyFFTEngine((8, 6), communicator=comm)
