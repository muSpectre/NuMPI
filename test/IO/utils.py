import numpy as np

from NuMPI.Testing.Subdivision import suggest_subdivisions


class DistributedData:
    def __init__(self, data, nb_domain_grid_pts, subdomain_locations):
        self.data = data
        self.nb_domain_grid_pts = nb_domain_grid_pts
        self.nb_subdomain_grid_pts = data.shape
        self.subdomain_locations = subdomain_locations

    @property
    def subdomain_slices(self):
        return tuple(
            slice(s, s + n)
            for s, n in zip(self.subdomain_locations, self.nb_subdomain_grid_pts)
        )


def get_coord(rank, subdivisions):
    coord = []
    for n in subdivisions:
        coord += [rank % n]
        rank = rank // n
    return coord


def subdivide(comm, globaldata):
    """
    Returns the part of the `globaldata` array distributed on a grid.

    Parameters
    ----------
    comm : communicator
        The MPI communicator.
    globaldata : numpy.ndarray
        The global data array to be decomposed.

    Returns
    -------
    DistributedData
        The part of the global data assigned to the current rank.
    """
    nb_domain_grid_pts = globaldata.shape
    subdivisions = suggest_subdivisions(len(nb_domain_grid_pts), comm.Get_size())
    coord = get_coord(comm.Get_rank(), subdivisions)
    nb_subdomain_grid_pts = np.array(globaldata.shape) // subdivisions

    subdomain_locations = tuple(n * c for n, c in zip(nb_subdomain_grid_pts, coord))

    nb_subdomain_grid_pts = tuple(
        n if c < s - 1 else m - n * (s - 1)
        for c, s, n, m in zip(
            coord, subdivisions, nb_subdomain_grid_pts, nb_domain_grid_pts
        )
    )

    subdomain_slices = tuple(
        slice(s, s + n) for s, n in zip(subdomain_locations, nb_subdomain_grid_pts)
    )

    return DistributedData(
        globaldata[subdomain_slices].copy(), nb_domain_grid_pts, subdomain_locations
    )


def make_2d_slab_x(comm, globaldata):
    """
    Returns the part of globaldata attribute to the present rank in 2D data
    decomposition.

    Parameters
    ----------
    comm : communicator
        The MPI communicator.
    globaldata : numpy.ndarray
        The global data array to be decomposed.

    Returns
    -------
    DistributedData
        The part of the global data assigned to the current rank.
    """
    nprocs = comm.Get_size()
    rank = comm.Get_rank()

    nb_domain_grid_pts = globaldata.shape

    step = nb_domain_grid_pts[0] // nprocs

    if rank == nprocs - 1:
        subdomain_slices = (slice(rank * step, None), slice(None, None))
        subdomain_locations = [rank * step, 0]
        # nb_subdomain_grid_pts = [nb_domain_grid_pts[0] - rank * step,
        #                          nb_domain_grid_pts[1]]
    else:
        subdomain_slices = (slice(rank * step, (rank + 1) * step), slice(None, None))
        subdomain_locations = [rank * step, 0]
        # nb_subdomain_grid_pts = [step, nb_domain_grid_pts[1]]

    return DistributedData(
        globaldata[subdomain_slices].copy(), nb_domain_grid_pts, subdomain_locations
    )


def make_2d_slab_y(comm, globaldata):
    nprocs = comm.Get_size()
    rank = comm.Get_rank()

    nb_domain_grid_pts = globaldata.shape

    step = nb_domain_grid_pts[1] // nprocs
    if rank == nprocs - 1:
        subdomain_slices = (slice(None, None), slice(rank * step, None))
        subdomain_locations = [0, rank * step]
        # nb_subdomain_grid_pts = [nb_domain_grid_pts[0],
        #                          nb_domain_grid_pts[1] - rank * step]
    else:
        subdomain_slices = (slice(None, None), slice(rank * step, (rank + 1) * step))
        subdomain_locations = [0, rank * step]
        # nb_subdomain_grid_pts = [nb_domain_grid_pts[1], step]

    return DistributedData(
        globaldata[subdomain_slices].copy(), nb_domain_grid_pts, subdomain_locations
    )
