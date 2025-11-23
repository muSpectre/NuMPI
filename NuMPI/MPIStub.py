#
# Copyright 2019-2020 Antoine Sanner
#           2020 k.o.haase@googlemail.com
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
Stub implementation of mpi4py. This is necessary to run a serial version of
NuMPI and dependent projects without an MPI installation.

Important note: This is at present *not* the complete API, but only includes
the features that we need in our projects. If you miss some functionality,
implement it and submit it back to us.
"""

from enum import Enum
from typing import Sequence, Optional

import numpy as np
from io import TextIOBase


# ## Layout
class Layout(Enum):
    ORDER_C = 0
    ORDER_F = 1


ORDER_C = Layout.ORDER_C
ORDER_F = Layout.ORDER_F


# ## Data types


class Datatype(object):
    """
    Because the datatype in Stub is solely for File I/O purpose, the class assumes the datatype is always 
    some sections (chunks) of a numpy array.
    """
    
    _numpy_type: np.dtype
    """Base element"""
    _chunk_positions: Sequence[int]
    """In unit of elements"""
    _chunk_sizes: Sequence[int]
    """In unit of elements"""
    _lower_bound: int
    """In unit of elements"""
    _upper_bound: int
    """In unit of elements"""

    def __init__(
            self, numpy_type: np.dtype, chunk_positions: Sequence[int],
            chunk_sizes: Sequence[int],
            lower_bound: Optional[int] = None, upper_bound: Optional[int] = None):
        self._numpy_type = numpy_type

        # Check
        assert len(chunk_positions) == len(chunk_sizes)
        try:
            # Sort such that chunk positions are monotonically increasing
            [chunk_positions, chunk_sizes] = zip(*sorted(zip(chunk_positions, chunk_sizes)))
            # Merge contiguous chunks for better I/O performance
            [chunk_positions, chunk_sizes] = self.merge_contiguous_chunks(chunk_positions, chunk_sizes)
        except ValueError:
            # when there is no chunk specified (empty lists)
            pass
        self._chunk_positions = chunk_positions
        self._chunk_sizes = chunk_sizes

        # Default value
        if lower_bound is None:
            try:
                lower_bound = self._chunk_positions[0]
            except IndexError:
                # when there is no chunk specified (empty lists)
                lower_bound = 0
        self._lower_bound = lower_bound

        # Default value
        if upper_bound is None:
            try:
                upper_bound = self._chunk_positions[-1] + self._chunk_sizes[-1]
            except IndexError:
                # when there is no chunk specified (empty lists)
                upper_bound = 0
        self._upper_bound = upper_bound

    @property
    def element_size(self):
        """Size of element in byte units."""
        return self._numpy_type.itemsize

    @property
    def type_size(self):
        """Extent of the type in byte units."""
        return (self._upper_bound - self._lower_bound) * self._numpy_type.itemsize

    def iterate_chunks(self):
        """Iterate (position, size) of chunks."""
        return zip(self._chunk_positions, self._chunk_sizes)

    @staticmethod
    def merge_contiguous_chunks(positions, sizes):
        """Fully merge contiguous chunks in a vectorized way.
        Assumes inputs are sorted by (starting) position and non-overlapping.
        """
        positions = np.asarray(positions)
        sizes = np.asarray(sizes)

        # Special case
        if positions.size == 0:
            return positions, sizes

        # Compute end positions
        ends = positions + sizes

        # Find where continuity breaks: ends[i] != positions[i+1]
        # This gives a boolean array of length N-1, so mark start of first chunk also as break
        is_break = np.concatenate(([True], ends[:-1] != positions[1:]))

        # Get indices at where continuity breaks
        idx_breaks = np.where(is_break)[0]

        # New (start) position is the position of the first in merged chunks
        merged_positions = positions[idx_breaks]

        # New size is the sum of sizes of merged chunks
        merged_sizes = np.add.reduceat(sizes, idx_breaks)

        return merged_positions, merged_sizes

    def Get_size(self):
        """Return the number of bytes occupied by entries in the datatype."""
        return np.multiply.reduce(self._chunk_sizes) * self._numpy_type.itemsize

    def Commit(self):
        pass

    def Free(self):
        pass

    free = Free

    def Create_contiguous(self, count):
        """Refer to MPI 5.0 Doc 5.1.2 Datatype Constructors MPI_TYPE_CONTIGUOUS"""
        # Special case
        if count == 0:
            return Datatype(self._numpy_type, [], [])

        # Add corresponding offsets for each repetition
        positions = np.tile(self._chunk_positions, count).reshape(count, -1)
        extent = self._upper_bound - self._lower_bound
        for i_count in range(count):
            positions[i_count] += i_count * extent

        # Sizes are simply repeated
        sizes = np.tile(self._chunk_sizes, count)

        return Datatype(self._numpy_type, positions.ravel(), sizes)

    def Create_vector(self, count, blocklength, stride):
        """Refer to MPI 5.0 Doc 5.1.2 Datatype Constructors MPI_TYPE_VECTOR"""
        # Special case
        if count == 0 or blocklength == 0:
            return Datatype(self._numpy_type, [], [])

        # Add corresponding offsets for each repetition
        positions = np.tile(self._chunk_positions, [count, blocklength]).reshape(count, blocklength, -1)
        extent = self._upper_bound - self._lower_bound
        for i_count in range(count):
            for i_block in range(blocklength):
                positions[i_count, i_block] += (i_count * stride + i_block) * extent

        # Sizes are simply repeated
        sizes = np.tile(self._chunk_sizes, count * blocklength)

        return Datatype(self._numpy_type, positions.ravel(), sizes)

    def Create_subarray(self, sizes, subsizes, starts, order):
        """Refer to MPI 5.0 Doc 5.1.3 Subarray Datatype Constructor MPI_TYPE_CREATE_SUBARRAY"""
        # check
        assert len(subsizes) == len(sizes)
        assert len(starts) == len(sizes)
        assert order in [ORDER_C, ORDER_F]

        # Special case
        if np.any(subsizes == 0):
            return Datatype(self._numpy_type, [], [])

        # More or less a multidimensional version of 'Create_vector', plus some extra offsets due to 'starts'
        positions = np.tile(self._chunk_positions, subsizes).reshape(*subsizes, -1)
        extent = self._upper_bound - self._lower_bound
        start_offsets = np.asarray(starts)

        # Compute strides
        if order is ORDER_F:
            strides = np.multiply.accumulate([1, *sizes])[:-1]
        else:  # order is ORDER_C
            strides = np.flip(np.multiply.accumulate(np.flip([*sizes, 1]))[:-1])

        # Add corresponding displacements for each repetition
        for indices in np.ndindex(subsizes):
            positions[indices] += np.dot(start_offsets + indices, strides) * extent

        # Sizes are simply repeated
        sizes = np.tile(self._chunk_sizes, np.multiply.reduce(subsizes))

        # For subarray, the lb and ub are the begin and end for the whole array
        lb = 0
        ub = np.multiply.reduce(sizes) * extent

        return Datatype(self._numpy_type, positions.ravel(), sizes, lb, ub) 


class BasicDatatype(Datatype):

    def __init__(self, name):
        super().__init__(np.dtype(name), [0], [1])

    def _end_of_block(self, position):
        return None, None


class Typedict(object):
    def __getitem__(self, item):
        return BasicDatatype(item)


_typedict = Typedict()
BYTE = _typedict["i1"]


# ## Operations


class Operations(Enum):
    MIN = 1
    MAX = 2
    SUM = 3
    PROD = 4
    LAND = 5
    BAND = 6
    LOR = 7
    BOR = 8
    LXOR = 9
    BXOR = 10
    MAXLOC = 11
    MINLOC = 12


MIN = Operations.MIN
MAX = Operations.MAX
SUM = Operations.SUM
PROD = Operations.PROD
LAND = Operations.LAND
BAND = Operations.BAND
LOR = Operations.LOR
BOR = Operations.BOR
LXOR = Operations.LXOR
BXOR = Operations.BXOR
MAXLOC = Operations.MAXLOC
MINLOC = Operations.MINLOC


# ## Opening modes


class OpeningMode(Enum):
    MODE_RDONLY = 1
    MODE_WRONLY = 2
    MODE_RDWR = 3
    MODE_CREATE = 4
    # FIXME: The following modes are not (yet) supported
    # MODE_EXCL = 8
    # MODE_DELETE_ON_CLOSE = 'A'
    # MODE_UNIQUE_OPEN = 'A'
    # MODE_SEQUENTIAL = 'A'
    # MODE_APPEND = 'A'

    _MODE_6 = 6

    def __or__(self, other):
        return OpeningMode(self.value | other.value)

    def std_mode(self):
        if self.MODE_CREATE.value & self.value:
            return "wb"
        if self.MODE_WRONLY.value & self.value:
            return "ab"
        return "rb"


MODE_RDONLY = OpeningMode.MODE_RDONLY
MODE_WRONLY = OpeningMode.MODE_WRONLY
MODE_RDWR = OpeningMode.MODE_RDWR
MODE_CREATE = OpeningMode.MODE_CREATE


# FIXME: The following modes are not (yet) supported
# MODE_EXCL = OpeningMode.MODE_EXCL
# MODE_DELETE_ON_CLOSE = OpeningModes.MODE_DELETE_ON_CLOSE
# MODE_UNIQUE_OPEN = OpeningModes.MODE_UNIQUE_OPEN
# MODE_SEQUENTIAL = OpeningModes.MODE_SEQUENTIAL
# MODE_APPEND = OpeningModes.MODE_APPEND


# ## Stub communicator object


class Intracomm(object):
    def Barrier(self):
        pass

    barrier = Barrier

    def Get_rank(self):
        return 0

    rank = property(Get_rank)

    def Get_size(self):
        return 1

    size = property(Get_size)

    def Reduce(self, sendbuf, recvbuf, op=Operations.SUM, root=0):
        if root != 0:
            raise ValueError("Root must be zero for MPI stub implementation.")

        try:
            senddata = sendbuf
            sendtype = sendbuf.dtype
        except AttributeError:
            senddata, sendtype = sendbuf

        try:
            recvdata = recvbuf
            recvtype = recvbuf.dtype
        except AttributeError:
            recvdata, recvtype = recvbuf

        if sendtype != recvtype:
            raise TypeError(
                "Mismatch in send and receive MPI data types in MPI stub implementation. "
                f"Send type is {sendtype} while receive type is {recvtype}."
            )

        recvdata[...] = senddata

    Allreduce = Reduce

    def Allgather(self, sendbuf, recvbuf_counts):
        try:
            senddata = sendbuf
            sendtype = sendbuf.dtype
        except AttributeError:
            senddata, sendtype = sendbuf

        try:
            recvdata, counts = recvbuf_counts
            recvtype = recvdata.dtype
        except (TypeError, AttributeError):
            recvdata, counts, recvtype = recvbuf_counts

        if sendtype != recvtype:
            raise TypeError(
                "Mismatch in send and receive MPI data types in MPI stub implementation. "
                f"Send type is {sendtype} while receive type is {recvtype}."
            )

        recvdata[...] = senddata

    Allgatherv = Allgather


# ## Stub file I/O object


class File(object):
    @classmethod
    def Open(cls, comm, filename, amode=MODE_RDONLY):
        # FIXME: This method has an optional info argument
        return File(comm, filename, amode)

    def __init__(self, comm, filename, amode):
        if not isinstance(comm, Intracomm):
            raise RuntimeError(
                "Communicator object must be an instance of `Intracomm`."
            )

        self.already_open = False
        if not hasattr(filename, "read"):
            self._file = open(filename, amode.std_mode())
        else:
            self.already_open = True
            if isinstance(filename, TextIOBase):
                self._file = filename.buffer
            else:
                self._file = filename
        self.Set_view()

    def Close(self):
        self._file.close()

    def Get_position(self):
        """Return the current position of the individual file pointer.
        Note: Position is measured in etype units relative to the current file view.
        """
        return (self._file.tell() - self._view_start) // self._etype.type_size

    def Read(self, buf):
        try:
            if self._filetype is None:
                self._file.seek(self._view_start)
                # read until the buffer is full
                nb_bytes = buf.size * buf.itemsize
                data = self._file.read(nb_bytes)
            else:
                element_size = self._filetype.element_size
                data = bytearray()
                for position, size in self._filetype.iterate_chunks():
                    self._file.seek(self._view_start + position * element_size)
                    data += self._file.read(size * element_size)
            buf[...] = np.frombuffer(data, dtype=buf.dtype, count=buf.size).reshape(
                buf.shape, order='F' if not buf.flags.c_contiguous else 'C')
            self._view_start = self._file.tell()
        except Exception:
            if not self.already_open:
                self.close()

    Read_all = Read

    def Set_view(self, disp=0, etype=BYTE, filetype=None):
        self._view_start = disp
        self._etype = etype
        self._filetype = filetype

    def Write(self, buf):
        if isinstance(buf, np.ndarray):
            buf = buf.tobytes(order='F' if not buf.flags.c_contiguous else 'C')
        if self._filetype is None:
            self._file.seek(self._view_start)
            self._file.write(buf)
        else:
            element_size = self._filetype.element_size
            i_byte = 0
            for position, size in self._filetype.iterate_chunks():
                self._file.seek(self._view_start + position * element_size)
                nb_bytes = size * element_size
                self._file.write(buf[i_byte : i_byte + nb_bytes])
                i_byte += nb_bytes
        self._view_start = self._file.tell()

    Write_all = Write


COMM_WORLD = Intracomm()
COMM_SELF = Intracomm()
