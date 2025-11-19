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
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from io import TextIOBase


# ## Layout
class Layout(Enum):
    ORDER_C = 0
    ORDER_F = 1


ORDER_C = Layout.ORDER_C
ORDER_F = Layout.ORDER_F


# ## Data types


@dataclass(init=True)
class Datatype(object):
    _type_sigatures: Sequence[np.dtype]
    _displacements: Sequence[int]

    def Get_size(self):
        """Return the number of bytes occupied by entries in the datatype."""
        return np.multiply.reduce([dtype.itemsize for dtype in self._type_sigatures])

    def Get_extent(self):
        """Return lower bound and extent of datatype."""
        lb = min(self._displacements)
        # FIXME: alignment?
        i_ub = int(np.argmax(self._displacements))
        ub = self._displacements[i_ub] + self._type_sigatures[i_ub].itemsize
        return lb, ub - lb

    def Commit(self):
        pass

    def Free(self):
        pass

    def Create_contiguous(self, count):
        """Refer to MPI 5.0 Doc 5.1.2 Datatype Constructors MPI_TYPE_CONTIGUOUS"""
        # Type signatures are simply repeated
        type_sigs = list(self._type_sigatures) * count

        # Add corresponding displacements for each repetition
        displs = np.tile(self._displacements, count).reshape(count, -1)
        [_, extent] = self.Get_extent()
        for i_count in range(count):
            displs[i_count] += i_count * extent
        return Datatype(type_sigs, displs.ravel().tolist())

    def Create_vector(self, count, blocklength, stride):
        """Refer to MPI 5.0 Doc 5.1.2 Datatype Constructors MPI_TYPE_VECTOR"""
        # Type signatures are simply repeated
        type_sigs = list(self._type_sigatures) * (count * blocklength)

        # Add corresponding displacements for each repetition
        displs = np.tile(self._displacements, [count, blocklength]).reshape(count, blocklength, -1)
        [_, extent] = self.Get_extent()
        for i_count in range(count):
            for i_block in range(blocklength):
                displs[i_count, i_block] += (i_count * stride + i_block) * extent
        return Datatype(type_sigs, displs.ravel().tolist())

    def Create_subarray(self, sizes, subsizes, starts, order):
        """Refer to MPI 5.0 Doc 5.1.3 Subarray Datatype Constructor MPI_TYPE_CREATE_SUBARRAY"""
        # check
        nb_dim = len(sizes)
        assert len(subsizes) == nb_dim
        assert len(starts) == nb_dim
        assert order in [ORDER_C, ORDER_F]

        # Type signatures are simply repeated
        type_sigs = list(self._type_sigatures) * np.multiply.reduce(subsizes)

        # More or less a multidimensional version of 'Create_vector', plus some extra offsets due to 'starts'
        displs = np.tile(self._displacements, subsizes).reshape(*subsizes, -1)
        [_, extent] = self.Get_extent()
        start_offsets = np.asarray(starts)

        # Compute strides
        if order is ORDER_F:
            strides = np.multiply.accumulate([1, *sizes])[:-1]
        else:  # order is ORDER_C
            strides = np.flip(np.multiply.accumulate(np.flip([*sizes, 1]))[:-1])

        # Add corresponding displacements for each repetition
        for indices in np.ndindex(subsizes):
            displs[indices] += np.dot(start_offsets + indices, strides) * extent

        return Datatype(type_sigs, displs.ravel().tolist()) 


class BasicDatatype(Datatype):

    def __init__(self, name):
        self._type_sigatures = [np.dtype(name)]
        self._displacements = [0]

    def _end_of_block(self, position):
        return None, None


class Typedict(object):
    def __getitem__(self, item):
        return BasicDatatype(item)


_typedict = Typedict()


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
        self._disp = 0
        self._etype = _typedict["i1"]
        self._filetype = None

    def Close(self):
        self._file.close()

    def Get_position(self):
        return self._file.tell() - self._disp

    def Read(self, buf):
        try:
            data = self._file.read(buf.size * buf.itemsize)
            buf[...] = np.frombuffer(data, count=buf.size, dtype=buf.dtype).reshape(
                buf.shape
            )
        except Exception:
            if not self.already_open:
                self.close()

    Read_all = Read

    def Set_view(self, disp, etype=None, filetype=None):
        self._file.seek(disp)
        self._disp = disp
        self._etype = etype
        self._filetype = filetype

    def Write(self, buf):
        self._file.write(buf)

    Write_all = Write


COMM_WORLD = Intracomm()
COMM_SELF = Intracomm()
