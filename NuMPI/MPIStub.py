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

import numpy as np
from io import TextIOBase


# ## Data types


class VectorDatatype(object):
    def __init__(self, oldtype, count, blocklength, stride):
        if stride != blocklength:
            raise NotImplementedError(
                "Only vector datatypes with stride == blocklength are "
                "presently supported by "
                "the MPI stub interface."
            )
        self._oldtype = oldtype
        self._count = count
        self._blocklength = blocklength
        self._stride = stride

    def Commit(self):
        pass

    def Free(self):
        pass

    def Get_size(self):
        return self._count * self._blocklength * self._oldtype.Get_size()

    def _get_oldtype(self):
        return self._oldtype


class Datatype(object):
    def __init__(self, name):
        self._dtype = np.dtype(name)

    def Create_vector(self, count, blocklength, stride):
        return VectorDatatype(self, count, blocklength, stride)

    def Create_contiguous(self, count):
        return VectorDatatype(self, count, 1, 1)

    def Create_subarray(self, sizes, subsizes, starts, order):
        assert len(sizes) == len(subsizes)
        # new_type = self
        # for [size, subsize] in zip(sizes, subsizes):
        #     new_type = new_type.Create_vector(1, subsize, size)
        # return new_type
        return VectorDatatype(self, 1, np.multiply.reduce(subsizes), np.multiply.reduce(sizes))

    def Get_size(self):
        return self._dtype.itemsize

    def _end_of_block(self, position):
        return None, None


class Typedict(object):
    def __getitem__(self, item):
        return Datatype(item)


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
