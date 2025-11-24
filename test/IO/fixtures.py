import os

import numpy as np
import pytest

from NuMPI.IO.common import recover_shape


@pytest.fixture
def npyfile():
    """
    defines a filename and makes a cleanup once the test was executed
    Yields
    ------
    filename

    """
    yield "test_same_numpy.npy"
    try:
        os.remove("test_same_numpy.npy")
    except FileNotFoundError:
        pass


@pytest.fixture(params=("C", "F"))
def globaldata2d(request, comm):
    order = request.param

    rank = comm.Get_rank()

    nb_domain_grid_pts = (128, 128)
    np.random.seed(2)
    if order == "C":
        data = np.random.random(nb_domain_grid_pts)
        assert data.flags["C_CONTIGUOUS"]
    elif order == "F":
        data = np.random.random(nb_domain_grid_pts[::-1]).transpose()
        assert data.flags["F_CONTIGUOUS"]
    if rank == 0:
        np.save("test_fileload_2d.npy", data)

    comm.barrier()

    yield data
    comm.barrier()
    if rank == 0:
        os.remove("test_fileload_2d.npy")


@pytest.fixture(params=("C", "F"))
def globaldata3d(request, comm):
    order = request.param

    rank = comm.Get_rank()

    nb_domain_grid_pts = (13, 27, 9)
    np.random.seed(2)
    if order == "C":
        data = np.random.random(nb_domain_grid_pts)
        assert data.flags.c_contiguous
    elif order == "F":
        data = np.random.random(nb_domain_grid_pts[::-1]).transpose()
        assert data.flags.f_contiguous
    if rank == 0:
        np.save("test_3d.npy", data)

    comm.barrier()

    yield data
    comm.barrier()
    if rank == 0:
        os.remove("test_3d.npy")


@pytest.fixture(
    params=[
        ("C", (2, 3)),
        ("F", (2, 3)),
        ("C", (13, 7)),
        ("F", (13, 7)),
        ("C", (7, 13)),
        ("F", (7, 13)),
        ("C", (13, 11, 27)),
        ("F", (13, 11, 27)),
        ("C", (15, 7, 9, 3)),
        ("F", (15, 7, 9, 3)),
    ]
)
def datagrid(request, comm):
    order, nb_domain_grid_pts = request.param

    np.random.seed(2)
    if order == "C":
        data = np.random.random(nb_domain_grid_pts)
        # data = np.arange(np.prod(nb_domain_grid_pts)).reshape(nb_domain_grid_pts)
        assert data.flags.c_contiguous
    elif order == "F":
        data = np.random.random(nb_domain_grid_pts[::-1]).transpose()
        # data = np.arange(np.prod(nb_domain_grid_pts)).reshape(nb_domain_grid_pts[::-1]).T
        assert data.flags.f_contiguous

    return data


@pytest.fixture(
    params=[
        # 2D grid
        ("C", (16, 16), (3,), True),
        ("F", (16, 16), (3,), True),
        ("C", (16, 16), (3,), False),
        ("F", (16, 16), (3,), False),
        ("C", (16, 16), (2, 2), True),
        ("F", (16, 16), (2, 2), True),
        ("C", (16, 16), (2, 2), False),
        ("F", (16, 16), (2, 2), False),
        # 3D grid
        ("C", (8, 8, 8), (3,), True),
        ("F", (8, 8, 8), (3,), True),
        ("C", (8, 8, 8), (3,), False),
        ("F", (8, 8, 8), (3,), False),
        ("C", (8, 8, 8), (2, 2), True),
        ("F", (8, 8, 8), (2, 2), True),
        ("C", (8, 8, 8), (2, 2), False),
        ("F", (8, 8, 8), (2, 2), False),
        # Edge cases when some process may have no elements
        ("C", (3, 2), (), True),
        ("C", (3, 2), (), False),
    ]
)
def multicomponent_globaldata(request, comm):
    """
    Fixture for multicomponent data (2D and 3D).
    Returns a tuple of (data, nb_components, components_are_leading)
    """
    order, nb_grid_pts, nb_components, components_are_leading = request.param

    full_shape = recover_shape(nb_grid_pts, nb_components, components_are_leading)

    np.random.seed(2)
    if order == "C":
        data = np.random.random(full_shape)
        assert data.flags["C_CONTIGUOUS"]
    elif order == "F":
        data = np.random.random(full_shape[::-1]).transpose()
        assert data.flags["F_CONTIGUOUS"]

    rank = comm.Get_rank()

    spatial_ndim = len(nb_grid_pts)
    filename = f"test_multicomp_{spatial_ndim}d_{order}_{'leading' if components_are_leading else 'trailing'}.npy"
    if rank == 0:
        np.save(filename, data)

    comm.barrier()
    yield data, nb_grid_pts, nb_components, components_are_leading

    comm.barrier()
    if rank == 0:
        os.remove(filename)
