import numpy as np

from NuMPI.IO.common import decompose_shape, recover_shape
from NuMPI.Testing.Subdivision import suggest_subdivisions


class DistributedData:
    def __init__(self, data, nb_domain_grid_pts, subdomain_locations, nb_components=(), components_are_leading=True):
        self.data = data
        self.nb_domain_grid_pts = nb_domain_grid_pts
        self.nb_subdomain_grid_pts = decompose_shape(data.shape, len(
            data.shape) - len(nb_components), components_are_leading)[0]
        self.subdomain_locations = subdomain_locations
        self.component_ndim = len(nb_components)
        self.components_are_leading = components_are_leading

    @property
    def subdomain_slices(self):
        spatial_slices = tuple(
            slice(s, s + n)
            for s, n in zip(self.subdomain_locations, self.nb_subdomain_grid_pts)
        )
        component_slices = (slice(None),) * self.component_ndim
        # Though these are slices, but the logic is the same.
        return recover_shape(spatial_slices, component_slices, self.components_are_leading)


def get_coord(rank, subdivisions):
    coord = []
    for n in subdivisions:
        coord += [rank % n]
        rank = rank // n
    return coord


def subdivide(comm, globaldata, nb_components=(), components_are_leading=True):
    """
    Returns the part of the `globaldata` array distributed on a grid.

    Parameters
    ----------
    comm : communicator
        The MPI communicator.
    globaldata : numpy.ndarray
        The global data array to be decomposed.
    nb_components: tuple[int,...]
        Shape of components.
    components_are_leading: bool
        If True, full shape is (*nb_components, *spaital_shape); else, (*spatial_shape, *nb_components)

    Returns
    -------
    DistributedData
        The part of the global data assigned to the current rank.
    """
    nb_domain_grid_pts = decompose_shape(
        globaldata.shape, len(globaldata.shape) - len(nb_components),
        components_are_leading)[0]
    subdivisions = suggest_subdivisions(len(nb_domain_grid_pts), comm.Get_size())
    coord = get_coord(comm.Get_rank(), subdivisions)
    nb_subdomain_grid_pts = np.array(nb_domain_grid_pts) // subdivisions

    subdomain_locations = tuple(n * c for n, c in zip(nb_subdomain_grid_pts, coord))

    nb_subdomain_grid_pts = tuple(
        n if c < s - 1 else m - n * (s - 1)
        for c, s, n, m in zip(
            coord, subdivisions, nb_subdomain_grid_pts, nb_domain_grid_pts
        )
    )

    spatial_slices = tuple(
        slice(s, s + n) for s, n in zip(subdomain_locations, nb_subdomain_grid_pts)
    )
    component_slices = (slice(None),) * len(nb_components)
    subdomain_slices = recover_shape(spatial_slices, component_slices, components_are_leading)

    return DistributedData(
        globaldata[subdomain_slices].copy(),
        nb_domain_grid_pts, subdomain_locations, nb_components, components_are_leading
    )


def make_slab_x(comm, globaldata, nb_components=(), components_are_leading=True):
    """
    Returns the part of globaldata attribute to the present rank in 2D data
    decomposition.

    Parameters
    ----------
    comm : communicator
        The MPI communicator.
    globaldata : numpy.ndarray
        The global data array to be decomposed.
    nb_components: tuple[int,...]
        Shape of components.
    components_are_leading: bool
        If True, full shape is (*nb_components, *spaital_shape); else, (*spatial_shape, *nb_components)

    Returns
    -------
    DistributedData
        The part of the global data assigned to the current rank.
    """
    nprocs = comm.Get_size()
    rank = comm.Get_rank()

    nb_domain_grid_pts = decompose_shape(
        globaldata.shape, len(globaldata.shape) - len(nb_components),
        components_are_leading)[0]

    step = nb_domain_grid_pts[0] // nprocs

    spatial_slices = [slice(None)] * len(nb_domain_grid_pts)
    if rank == nprocs - 1:
        spatial_slices[0] = slice(rank * step, None)
    else:
        spatial_slices[0] = slice(rank * step, (rank + 1) * step)
    component_slices = (slice(None),) * len(nb_components)
    subdomain_slices = recover_shape(tuple(spatial_slices), component_slices, components_are_leading)

    subdomain_locations = [0] * len(nb_domain_grid_pts)
    subdomain_locations[0] = rank * step

    return DistributedData(
        globaldata[subdomain_slices].copy(), nb_domain_grid_pts, subdomain_locations
    )


def make_slab_y(comm, globaldata, nb_components=(), components_are_leading=True):
    nprocs = comm.Get_size()
    rank = comm.Get_rank()

    nb_domain_grid_pts = decompose_shape(
        globaldata.shape, len(globaldata.shape) - len(nb_components),
        components_are_leading)[0]

    if len(nb_domain_grid_pts) < 1:
        raise ValueError("The domain must be at least 2D to have a y-axis.")

    step = nb_domain_grid_pts[1] // nprocs

    spatial_slices = [slice(None)] * len(nb_domain_grid_pts)
    if rank == nprocs - 1:
        spatial_slices[1] = slice(rank * step, None)
    else:
        spatial_slices[1] = slice(rank * step, (rank + 1) * step)
    component_slices = (slice(None),) * len(nb_components)
    subdomain_slices = recover_shape(tuple(spatial_slices), component_slices, components_are_leading)

    subdomain_locations = [0] * len(nb_domain_grid_pts)
    subdomain_locations[1] = rank * step

    return DistributedData(
        globaldata[subdomain_slices].copy(), nb_domain_grid_pts, subdomain_locations
    )
