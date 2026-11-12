#
# Copyright 2018, 2020 Antoine Sanner
#           2019 Lars Pastewka
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
import numpy.ma as ma
import pytest

from NuMPI.Tools import Reduction


@pytest.fixture
def pnp(comm):
    return Reduction(comm)


def test_sum_scalar(pnp):
    res = pnp.sum(np.array(1))
    assert res == pnp.comm.Get_size()


@pytest.mark.parametrize('arr', [
    [1, 2.1, 3],
    np.array((1, 2.1, 3)),
    ma.masked_array([1, 2.1, 3]),
    ma.masked_array([2, 3.1, 3], mask=[True, False, False]),
])
def test_sum_1D(pnp, arr):
    res = pnp.sum(arr)
    np.testing.assert_allclose(res, pnp.comm.Get_size() * 6.1, atol=1e-12)


@pytest.mark.parametrize('arr', [
    [[1, 2.1, 3],
     [4, 5, 6]],
    np.array(((1, 2.1, 3),
              (4, 5, 6))),
    ma.masked_array([[1, 2.1, 3],
                     [4, 5, 6]])
])
def test_sum_2D(pnp, arr):
    res = pnp.sum(arr)
    np.testing.assert_allclose(res, pnp.comm.Get_size() * 21.1, atol=1e-12)


@pytest.mark.parametrize('arr', [
    np.array(((1, 2.1, 3),
              (4, 5, 6))),
    ma.masked_array([[1, 2.1, 3],
                     [4, 5, 6]])
])
def test_sum_boolean(pnp, arr):
    arr = arr > 3

    # print(arr.dtype)
    res = pnp.sum(arr)
    assert res == pnp.comm.Get_size() * 3


def test_sum_along_axis_decomp(pnp):
    nb_domain_grid_pts = (128, 65)
    np.random.seed(2)
    globaldata = np.random.random(nb_domain_grid_pts)

    nprocs = pnp.comm.Get_size()
    rank = pnp.comm.Get_rank()

    step = nb_domain_grid_pts[0] // nprocs

    if rank == nprocs - 1:
        subdomain_slices = (slice(rank * step, None), slice(None, None))
        # subdomain_locations = [rank * step, 0]
        # nb_subdomain_grid_pts = [nb_domain_grid_pts[0] - rank * step,
        #                         nb_domain_grid_pts[1]]
    else:
        subdomain_slices = (
            slice(rank * step, (rank + 1) * step), slice(None, None))
        # subdomain_locations = [rank * step, 0]
        # nb_subdomain_grid_pts = [step, nb_domain_grid_pts[1]]

    localdata = globaldata[subdomain_slices]

    np.testing.assert_allclose(pnp.sum(localdata, axis=0),
                               np.sum(globaldata, axis=0), rtol=1e-12)
    # Why are they not perfectly equal ?


@pytest.mark.parametrize('arr', [
    np.reshape(np.array((-1, 1, 5, 4,
                         4, 5, 4, 5,
                         7, 0, 1, 0.), dtype=float), (3, 4)),
    ma.masked_array(np.reshape(np.array((-1, 1, 5, 4,
                                         4, 5, 4, 5,
                                         7, 0, 1, 0.), dtype=float), (3, 4)))
])
def test_max_2D(pnp, arr):
    rank = pnp.comm.Get_rank()
    if pnp.comm.Get_size() >= 4:
        if rank == 0:
            local_arr = arr[0:2, 0:2]
        elif rank == 1:
            local_arr = arr[0:2, 2:]
        elif rank == 2:
            local_arr = arr[2:, 0:2]
        elif rank == 3:
            local_arr = arr[2:, 2:]
        else:
            local_arr = np.empty(0, dtype=arr.dtype)
    elif pnp.comm.Get_size() >= 2:
        if rank == 0:
            local_arr = arr[0:2, :]
        elif rank == 1:
            local_arr = arr[2:, :]
        else:
            local_arr = np.empty(0, dtype=arr.dtype)
    else:
        local_arr = arr
    assert pnp.max(local_arr) == 7


@pytest.mark.parametrize('arr', [
    np.reshape(np.array((-1, 1, 5, 4,
                         4, 5, 4, 5,
                         7, 0, 1, 0), dtype=int), (3, 4)),
    ma.masked_array(np.reshape(np.array((-1, 1, 5, 4,
                                         4, 5, 4, 5,
                                         7, 0, 1, 0), dtype=int), (3, 4)))
])
def test_max_2D_int(pnp, arr):
    rank = pnp.comm.Get_rank()
    if pnp.comm.Get_size() >= 4:
        if rank == 0:
            local_arr = arr[0:2, 0:2]
        elif rank == 1:
            local_arr = arr[0:2, 2:]
        elif rank == 2:
            local_arr = arr[2:, 0:2]
        elif rank == 3:
            local_arr = arr[2:, 2:]
        else:
            local_arr = np.empty(0, dtype=arr.dtype)
    elif pnp.comm.Get_size() >= 2:
        if rank == 0:
            local_arr = arr[0:2, :]
        elif rank == 1:
            local_arr = arr[2:, :]
        else:
            local_arr = np.empty(0, dtype=arr.dtype)
    else:
        local_arr = arr
    assert pnp.max(local_arr) == 7


def test_max_min_empty(pnp):
    """
    Sometimes the input array is empty
    """
    if pnp.comm.Get_size() >= 2:
        if pnp.comm.Get_rank() == 0:
            local_arr = np.array([], dtype=float)

        else:
            local_arr = np.array([1, 0, 4], dtype=float)
        assert pnp.max(local_arr) == 4
        assert pnp.min(local_arr) == 0

        if pnp.comm.Get_rank() == 0:
            local_arr = np.array([1, 0, 4], dtype=float)
        else:

            local_arr = np.array([], dtype=float)
        assert pnp.max(local_arr) == 4
        assert pnp.min(local_arr) == 0

    else:
        local_arr = np.array([], dtype=float)
        # self.assertTrue(np.isnan(pnp.max(local_arr)))
        # self.assertTrue(np.isnan(pnp.min(local_arr)))
        assert pnp.max(local_arr) == np.finfo(local_arr.dtype).min
        assert pnp.min(local_arr) == np.finfo(local_arr.dtype).max


@pytest.mark.parametrize('arr', [
    np.reshape(np.array((-1, 1, 5, 4,
                         4, 5, 4, 5,
                         7, 0, 1, 0), dtype=float), (3, 4)),
    ma.masked_array(np.reshape(np.array((-1, 1, 5, 4,
                                         4, 5, 4, 5,
                                         7, 0, 1, 0), dtype=float), (3, 4)))
])
def test_min(pnp, arr):
    rank = pnp.comm.Get_rank()
    if pnp.comm.Get_size() >= 4:
        if rank == 0:
            local_arr = arr[0:2, 0:2]
        elif rank == 1:
            local_arr = arr[0:2, 2:]
        elif rank == 2:
            local_arr = arr[2:, 0:2]
        elif rank == 3:
            local_arr = arr[2:, 2:]
        else:
            local_arr = np.empty(0, dtype=arr.dtype)
    elif pnp.comm.Get_size() >= 2:
        if rank == 0:
            local_arr = arr[0:2, :]
        elif rank == 1:
            local_arr = arr[2:, :]
        else:
            local_arr = np.empty(0, dtype=arr.dtype)
    else:
        local_arr = arr
    assert pnp.min(local_arr) == -1


def test_dot_row_vectors(pnp):
    np.random.seed(1)

    n = 10
    fulla = np.random.random(n)
    fullb = np.random.random(n)

    worldsize = pnp.comm.Get_size()
    rank = pnp.comm.Get_rank()
    step = n // worldsize

    if rank == worldsize - 1:
        loc_sl = slice(rank * step, None)
    else:
        loc_sl = slice(rank * step, (rank + 1) * step)

    # because the addition of floating point numbers are not done in the same
    # order due to the parallelization, the truncation errors can be different
    # TODO: maybe we can quantify more seriously what the error should be ?
    np.testing.assert_almost_equal(pnp.dot(fulla[loc_sl], fullb[loc_sl]),
                                   np.dot(fulla, fullb), 7)


def test_dot_matrix_vector(pnp):
    np.random.seed(1)

    m = 13
    n = 11

    worldsize = pnp.comm.Get_size()
    rank = pnp.comm.Get_rank()

    step = n // worldsize

    if rank == worldsize - 1:
        loc_sl = slice(rank * step, None)
    else:
        loc_sl = slice(rank * step, (rank + 1) * step)

    fulla = np.random.random((m, n))
    for fullb in [np.random.random((n, 1)), np.random.random(n)]:
        np.testing.assert_allclose(pnp.dot(fulla[:, loc_sl], fullb[loc_sl]),
                                   np.dot(fulla, fullb))
        np.testing.assert_allclose(
            pnp.dot(fullb[loc_sl].T, fulla[:, loc_sl].T),
            np.dot(fullb.T, fulla.T))
        with pytest.raises(ValueError):
            pnp.dot(fullb[loc_sl], fulla[:, loc_sl])
        with pytest.raises(ValueError):
            pnp.dot(fulla[:, loc_sl].T, fullb[loc_sl])


def test_dot_matrix_matrix(pnp):
    np.random.seed(1)  # important so that every processor sees the same data

    worldsize = pnp.comm.Get_size()
    rank = pnp.comm.Get_rank()

    m = 5
    n = 10
    fulla = np.random.random((m, n))
    fullb = np.random.random((n, m))

    step = n // worldsize

    if rank == worldsize - 1:
        loc_sl = slice(rank * step, None)
    else:
        loc_sl = slice(rank * step, (rank + 1) * step)

    np.testing.assert_allclose(pnp.dot(fulla[:, loc_sl], fullb[loc_sl, :]),
                               np.dot(fulla, fullb))


def test_any_scalar(pnp):
    rank = pnp.comm.Get_rank()

    locval = False

    if rank == 0:
        locval = True

    assert pnp.any(locval)

    locval = False

    assert not pnp.any(locval)


@pytest.mark.parametrize('locval', [
    np.array([False, False, True]),
    ma.masked_array([False, False, True])
])
def test_any_array(pnp, locval):
    rank = pnp.comm.Get_rank()

    if rank == 0:
        locval = np.array([False, True])

    assert pnp.any(locval)

    locval = np.array([False, False, False])

    assert not pnp.any(locval)


def test_all_scalar(pnp):
    rank = pnp.comm.Get_rank()

    locval = True

    if rank == 0:
        locval = False

    assert not pnp.all(locval)

    locval = True

    assert pnp.all(locval)


@pytest.mark.parametrize('locval', [
    np.array([True, True, True]),
    ma.masked_array([True, True, True])
])
def test_all_array(pnp, locval):
    rank = pnp.comm.Get_rank()

    if rank == 0:
        locval = np.array([False, True])

    assert not pnp.all(locval)

    locval = np.array([True, True, True])

    assert pnp.all(locval)
