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


"""
MPI-parallel writing of arrays in numpy's 'npy' format.
"""

import struct
from ast import literal_eval
import contextlib
from typing import Sequence, Union

import numpy as np
from numpy.lib.format import MAGIC_PREFIX, magic

try:
    from numpy.lib.format import _filter_header
except ImportError:
    # Moved here in numpy 2.3
    from numpy.lib._format_impl import _filter_header

from .. import MPI
from ..Tools import Reduction
from .common import MPIFileTypeError, MPIFileView, decompose_shape, recover_shape


@contextlib.contextmanager
def cast_mpi_types(
        numpy_dtype: np.dtype, nb_grid_pts: Sequence[int],
        nb_subdomain_grid_pts: Sequence[int],
        subdomain_locations: Sequence[int],
        nb_components: Union[int, Sequence[int]],
        fortran_order: bool, components_are_leading: bool):

    # get the corresponding MPI Datatype
    elementary_type = MPI._typedict[numpy_dtype.char]
    if np.multiply.reduce(nb_subdomain_grid_pts) == 0:
        # For the process that gets zero elements, cast a filetype that eventually doesn't read / write
        file_type = elementary_type.Create_contiguous(0)
    else:
        # The domain decomposition only happens in spatial dimensions, so only the total size matters
        components_size = np.multiply.reduce(nb_components, dtype=int)
        # Either fortran order & components leading, or C order & components trailing -> components are contiguous
        components_are_contiguous = fortran_order ^ (not components_are_leading)
        if components_are_contiguous:
            # when component dimensions are contiguous, repeat the etype so it treats all components as a whole
            elementary_type = elementary_type.Create_contiguous(components_size)
        # Each process gets its own share of file according to the domain decomposition
        file_type = elementary_type.Create_subarray(
            nb_grid_pts, nb_subdomain_grid_pts, subdomain_locations, MPI.ORDER_F if fortran_order else MPI.ORDER_C)
        if not components_are_contiguous:
            # when spatial dimensions are contiguous, repeat the share-scheme for each component
            file_type = file_type.Create_contiguous(components_size)

    # Use context to guarantee that the types will be freed
    elementary_type.Commit()
    file_type.Commit()
    try:
        yield elementary_type, file_type
    finally:
        elementary_type.free()
        file_type.free()


class NPYFile(MPIFileView):
    """
    You may have a look at numpy.lib.format if you want to understand how
    this code works
    """

    def __init__(self, fn, comm):
        super().__init__(fn, comm)
        self._read_header()

    def detect_format(self):  # TODO: maybe useless
        try:
            self._read_header()
            return True
        except Exception:
            return False

    def _read_header(self):
        self.file = None
        try:
            magic_str = magic(1, 0)
            self.file = MPI.File.Open(self.comm, self.fn, MPI.MODE_RDONLY)
            magic_str = mpi_read_bytes(self.file, len(magic_str))
            if magic_str[:-2] != MAGIC_PREFIX:
                raise MPIFileTypeError(
                    "MAGIC_PREFIX missing at the beginning of file {}".format(self.fn)
                )

            version = magic_str[-2:]

            if version == b"\x01\x00":
                hlength_type = "<H"
            elif version == b"\x02\x00":
                hlength_type = "<I"
            else:
                raise MPIFileTypeError("Invalid version %r" % version)

            hlength_str = mpi_read_bytes(self.file, struct.calcsize(hlength_type))
            self.header_length = struct.unpack(hlength_type, hlength_str)[0]
            header = mpi_read_bytes(self.file, self.header_length)

            header = _filter_header(header.decode("latin1"))
            d = literal_eval(header)  # TODO: Copy from _read_array_header  with all the
            # assertions
            self.dtype = np.dtype(d["descr"])
            self.fortran_order = d["fortran_order"]
            self.array_shape = d["shape"]
            self.data_start = self.file.Get_position()

            self.header_length += len(magic_str) + struct.calcsize(hlength_type)
        except Exception as err:
            # FIXME! This should be handled through a resource manager
            if self.file is not None and self.close_file_on_error:
                self.file.Close()
            raise err

    def read(
            self, subdomain_locations: Union[Sequence[int], None] = None,
            nb_subdomain_grid_pts: Union[Sequence[int], None] = None,
            components_are_leading=True):
        # Figure out how many dimensions are spatial
        try:
            spatial_ndim = len(subdomain_locations)
        except TypeError:
            try:
                spatial_ndim = len(nb_subdomain_grid_pts)
            except TypeError:
                # take all the dimensions as spatial
                spatial_ndim = len(self.array_shape)
        # Get two part of shapes from the total shape
        nb_grid_pts, nb_components = decompose_shape(self.array_shape, spatial_ndim, components_are_leading)

        # If not specified, starting at the origin
        if subdomain_locations is None:
            subdomain_locations = (0,) * spatial_ndim
        # Check value compatibility
        assert len(subdomain_locations) == spatial_ndim

        # If not specified, take the whole grid
        if nb_subdomain_grid_pts is None:
            nb_subdomain_grid_pts = nb_grid_pts
        # Check value compatibility
        assert len(nb_subdomain_grid_pts) == spatial_ndim

        buf_shape = recover_shape(nb_subdomain_grid_pts, nb_components, components_are_leading)
        data = np.empty(buf_shape, dtype=self.dtype, order='F' if self.fortran_order else 'C')
        with cast_mpi_types(data.dtype, nb_grid_pts, nb_subdomain_grid_pts, subdomain_locations, nb_components,
                            self.fortran_order, components_are_leading) as [etype, filetype]:
            self.file.Set_view(self.header_length, etype, filetype)
            self.file.Read_all(data)
        return data

    def close(self):
        self.file.Close()


def mpi_open(fn, comm, format=None):
    readers = {"npy": NPYFile}

    if format is not None:
        try:
            reader = readers[format]
        except KeyError:
            raise (
                ValueError(
                    "Given format is not recognised, you should give {}".format(
                        readers.keys()
                    )
                )
            )
        return reader(fn, comm)

    for reader in readers.values():
        try:
            return reader(fn, comm)
        except MPIFileTypeError:
            pass
    raise MPIFileTypeError(
        "No MPI filereader was able to open_topography the file {}".format(fn)
    )


def mpi_read_bytes(file, nbytes):
    # allocate the buffer
    buf = np.empty(nbytes, dtype=np.int8)
    file.Read_all(buf)
    return buf.tobytes()


def save_npy(
        fn, data, subdomain_locations: Union[Sequence[int], None] = None,
        nb_grid_pts: Union[Sequence[int], None] = None,
        components_are_leading=True, comm=MPI.COMM_WORLD):
    """

    Parameters
    ----------
    data : numpy array : data owned by the processor
    location : index of the first element of data within the global data
    nb_grid_pts : nb_grid_pts of the global data
    comm : MPI communicator

    Returns
    -------

    """
    data = np.asarray(data)

    # Idiot check: Data cannot be strided
    if not data.flags.f_contiguous and not data.flags.c_contiguous:
        raise ValueError("Data must be contiguous")

    # Figure out how many dimensions are spatial
    try:
        spatial_ndim = len(subdomain_locations)
    except TypeError:
        try:
            spatial_ndim = len(nb_grid_pts)
        except TypeError:
            # take all the dimensions as spatial
            spatial_ndim = len(data.shape)
    # Get two type of shapes from buffer shape
    nb_subdomain_grid_pts, nb_components = decompose_shape(data.shape, spatial_ndim, components_are_leading)

    # If not specified, starting at the origin
    if subdomain_locations is None:
        subdomain_locations = (0,) * spatial_ndim
    # Check value compatibility
    assert len(subdomain_locations) == spatial_ndim

    # If not specified, what provided is whole grid
    if nb_grid_pts is None:
        nb_grid_pts = nb_subdomain_grid_pts
    # Check value compatibility
    assert len(nb_grid_pts) == spatial_ndim

    from numpy.lib.format import dtype_to_descr, magic

    # This is the NPY file version
    magic_str = magic(1, 0)

    # This indicates whether the data is written in "Fortran" order (column-major, i.e.
    # the first index is consecutive in memory). Since we do not want to copy the data
    # buffer, we just write to the file in the storage order that we find in memory.
    # We need to check across all MPI processes as two-dimension arrays where one
    # dimension has length unity report as both C-contiguous and F-contiguous.
    # (The storage order is ambiguous in this case.)
    fortran_order = bool(
        Reduction(comm).any(data.flags.f_contiguous and not data.flags.c_contiguous)
    )

    # This is the NPY file header - a stringified dict
    arr_dict_str = str(
        {
            "descr": dtype_to_descr(data.dtype),
            "fortran_order": fortran_order,
            "shape": recover_shape(nb_grid_pts, nb_components, components_are_leading),
        }
    )

    # Pad header until aligned (required by NPY format)
    while (len(arr_dict_str) + len(magic_str) + 2) % 16 != 15:
        arr_dict_str += " "
    arr_dict_str += "\n"
    header_length = len(arr_dict_str) + len(magic_str) + 2

    # Open for writing and dump header into file
    file = MPI.File.Open(comm, fn, MPI.MODE_CREATE | MPI.MODE_WRONLY)
    if comm.rank == 0:
        file.Write(magic_str)
        file.Write(np.int16(len(arr_dict_str)))
        file.Write(arr_dict_str.encode("latin-1"))

    # Write data
    with cast_mpi_types(data.dtype, nb_grid_pts, nb_subdomain_grid_pts, subdomain_locations,
                        nb_components, fortran_order, components_are_leading) as [etype, filetype]:
        file.Set_view(header_length, etype, filetype)
        file.Write_all(data)

    # Close file
    file.Close()


def load_npy(
        fn, subdomain_locations: Union[Sequence[int], None] = None,
        nb_subdomain_grid_pts: Union[Sequence[int], None] = None,
        components_are_leading=True, comm=MPI.COMM_WORLD):
    file = NPYFile(fn, comm)
    data = file.read(subdomain_locations, nb_subdomain_grid_pts, components_are_leading)
    file.close()
    return data
