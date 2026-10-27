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
Reading and writing .npy files with data types of either byte order. mpi4py
only accepts buffers in native byte order, so non-native files are a separate
code path; these tests run both with mpi4py and with the MPI stub. The same
holds for data types the MPI library does not provide.
"""

import os
from test.IO.utils import make_slab_x, make_slab_y

import numpy as np
import pytest

import NuMPI
from NuMPI import MPI
from NuMPI.IO.NPY import load_npy, save_npy

# Both byte orders explicitly, so that the non-native one is covered on any
# platform, and types without byte order
DTYPES = [byte_order + kind for kind in ["f8", "f4", "f2", "i8", "i2", "u2", "c16"] for byte_order in "<>"] + \
    ["u1", "i1"]


def _filename(comm, prefix, dtype, order, decompfun):
    # All ranks must agree on the file name
    byte_order = {"<": "le", ">": "be", "|": "na", "=": "ne"}[np.dtype(dtype).str[0]]
    return f"{prefix}_{comm.size}_{byte_order}{np.dtype(dtype).str[1:]}_{order}_{decompfun.__name__}.npy"


def _globaldata(dtype, order):
    data = np.arange(7 * 5).reshape(7, 5).astype(dtype)
    return np.asfortranarray(data) if order == "F" else np.ascontiguousarray(data)


def _remove(comm, fn):
    comm.barrier()
    if comm.rank == 0:
        os.remove(fn)


@pytest.mark.parametrize("decompfun", [make_slab_x, make_slab_y])
@pytest.mark.parametrize("order", ["C", "F"])
@pytest.mark.parametrize("dtype", DTYPES)
def test_load_byte_order(comm, dtype, order, decompfun):
    globaldata = _globaldata(dtype, order)
    distdata = decompfun(comm, globaldata)
    fn = _filename(comm, "test_load", dtype, order, decompfun)
    if comm.rank == 0:
        np.save(fn, globaldata)
    comm.barrier()

    loaded_data = load_npy(fn, subdomain_locations=distdata.subdomain_locations,
                           nb_subdomain_grid_pts=distdata.data.shape, comm=comm)

    # Same data type (including byte order) as numpy's load
    assert loaded_data.dtype == np.dtype(dtype)
    np.testing.assert_array_equal(loaded_data, distdata.data)
    _remove(comm, fn)


@pytest.mark.parametrize("decompfun", [make_slab_x, make_slab_y])
@pytest.mark.parametrize("order", ["C", "F"])
@pytest.mark.parametrize("dtype", DTYPES)
def test_save_byte_order(comm, dtype, order, decompfun):
    globaldata = _globaldata(dtype, order)
    distdata = decompfun(comm, globaldata)
    fn = _filename(comm, "test_save", dtype, order, decompfun)

    save_npy(fn, distdata.data, subdomain_locations=distdata.subdomain_locations,
             nb_grid_pts=distdata.nb_domain_grid_pts, comm=comm)
    comm.barrier()

    if comm.rank == 0:
        loaded_data = np.load(fn)
        assert loaded_data.dtype == np.dtype(dtype)
        np.testing.assert_array_equal(loaded_data, globaldata)
    _remove(comm, fn)


@pytest.mark.skipif(not NuMPI._has_mpi4py, reason="The MPI stub knows all data types")
@pytest.mark.parametrize("decompfun", [make_slab_x, make_slab_y])
@pytest.mark.parametrize("dtype", ["<f8", ">f8", "<i2", ">c16"])
def test_data_type_unknown_to_mpi(comm, dtype, decompfun, monkeypatch):
    # MPI libraries do not necessarily provide all data types (e.g. half
    # precision); such types are transferred as raw bytes. Hide the type from
    # mpi4py to test this independently of the MPI library.
    monkeypatch.delitem(MPI._typedict, np.dtype(dtype).char)
    globaldata = _globaldata(dtype, "C")
    distdata = decompfun(comm, globaldata)
    fn = _filename(comm, "test_unknown", dtype, "C", decompfun)

    save_npy(fn, distdata.data, subdomain_locations=distdata.subdomain_locations,
             nb_grid_pts=distdata.nb_domain_grid_pts, comm=comm)
    comm.barrier()
    if comm.rank == 0:
        np.testing.assert_array_equal(np.load(fn), globaldata)

    loaded_data = load_npy(fn, subdomain_locations=distdata.subdomain_locations,
                           nb_subdomain_grid_pts=distdata.data.shape, comm=comm)
    assert loaded_data.dtype == np.dtype(dtype)
    np.testing.assert_array_equal(loaded_data, distdata.data)
    _remove(comm, fn)
