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

Supported MPI Operations
========================

Communicator Methods (Intracomm)
---------------------------------
- ``Barrier()`` / ``barrier`` - No-op for single process
- ``Get_rank()`` / ``rank`` property - Always returns 0
- ``Get_size()`` / ``size`` property - Always returns 1
- ``Reduce(sendbuf, recvbuf, op, root)`` - Copies sendbuf to recvbuf (root must be 0)
- ``Allreduce(sendbuf, recvbuf, op)`` - Alias for Reduce
- ``Bcast(buf, root)`` / ``bcast`` - Broadcast (no-op, data already present)
- ``Gather(sendbuf, recvbuf, root)`` / ``gather`` - Gather to root (copies data)
- ``Gatherv(sendbuf, recvbuf, root)`` / ``gatherv`` - Gather with variable sizes
- ``Allgather(sendbuf, recvbuf)`` - Copies sendbuf to recvbuf
- ``Allgatherv(sendbuf, recvbuf)`` - Alias for Allgather
- ``Scatter(sendbuf, recvbuf, root)`` / ``scatter`` - Scatter from root
- ``Scatterv(sendbuf, recvbuf, root)`` / ``scatterv`` - Scatter with variable sizes

File I/O (File)
---------------
- ``Open(comm, filename, amode, info=None)`` - Open file for MPI I/O
- ``Close()`` - Close file
- ``Read(buf)`` / ``Read_all(buf)`` - Read data with view support
- ``Write(buf)`` / ``Write_all(buf)`` - Write data with view support
- ``Set_view(disp, etype, filetype)`` - Set file view with displacement and datatype
- ``Get_position()`` - Get file pointer position in etype units

Datatypes (Datatype)
--------------------
- ``Create_contiguous(count)`` - Create contiguous datatype (MPI-5.0 §5.1.2)
- ``Create_vector(count, blocklength, stride)`` - Create vector datatype (MPI-5.0 §5.1.2)
- ``Create_subarray(sizes, subsizes, starts, order)`` - Create subarray datatype (MPI-5.0 §5.1.3)
- ``Commit()`` - Commit datatype (no-op in stub)
- ``Free()`` / ``free()`` - Free datatype (no-op in stub)
- ``Get_size()`` - Get datatype size in bytes

Constants
---------
- **Operations:** MIN, MAX, SUM, PROD, LAND, BAND, LOR, BOR, LXOR, BXOR, MAXLOC, MINLOC
- **File modes:** MODE_RDONLY, MODE_WRONLY, MODE_RDWR, MODE_CREATE, MODE_EXCL, MODE_APPEND,
  MODE_DELETE_ON_CLOSE, MODE_UNIQUE_OPEN, MODE_SEQUENTIAL
- **Layouts:** ORDER_C, ORDER_F
- **Communicators:** COMM_WORLD, COMM_SELF (both single-process)

Unsupported Operations
======================

The following MPI operations are **NOT** implemented in the stub:

Point-to-Point Communication
-----------------------------
Send, Recv, Isend, Irecv, Sendrecv, Sendrecv_replace, Probe, Iprobe, Bsend, Ssend, Rsend

**Note:** Attempting to use Send/Recv/Isend/Irecv/Sendrecv/Probe/Iprobe will raise
``NotImplementedError`` with a helpful message suggesting alternatives.

Collective Operations (Missing)
--------------------------------
Alltoall, Alltoallv, Alltoallw, Reduce_scatter, Reduce_scatter_block, Scan, Exscan

Advanced Features
-----------------
- Request objects and non-blocking operation management (Wait, Test, Waitall, etc.)
- Communicator management (Dup, Split, Create, Free, Compare)
- Group operations (entire Group class)
- Topology support (Cartesian, Graph communicators)
- Distributed graph topology
- Neighborhood collectives
- One-sided communication (Put, Get, Accumulate, windows)
- Dynamic process management (Spawn, Connect, Accept)

If you need these features, please install mpi4py or contribute implementations.

Usage Notes
===========

**Root Rank Restriction**
    All collective operations that take a ``root`` parameter only accept ``root=0``
    in the stub, since there's only one process. Operations will raise ValueError
    if root != 0.

**Type Safety**
    The stub validates that send and receive buffers have matching dtypes and raises
    TypeError on mismatch.

**File I/O**
    File I/O operations support both file paths (as strings) and file-like objects.
    Note that file-like object support is stub-specific and not available in mpi4py.

**Serial Semantics**
    All operations have trivial single-process semantics:
    - Barriers are no-ops
    - Reductions copy data (no actual reduction)
    - Collectives involving all processes simply copy local data

Example Usage
=============

Basic communicator usage::

    from NuMPI import MPI

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()  # Always 0 with stub
    size = comm.Get_size()  # Always 1 with stub

Collective reduction::

    import numpy as np
    from NuMPI import MPI

    # This works with both stub and mpi4py
    sendbuf = np.array([1.0, 2.0, 3.0])
    recvbuf = np.zeros(3)
    MPI.COMM_WORLD.Allreduce(sendbuf, recvbuf, op=MPI.SUM)
    # With stub: recvbuf == sendbuf (single process, no reduction needed)

File I/O with datatypes::

    import numpy as np
    from NuMPI import MPI

    comm = MPI.COMM_WORLD
    data = np.arange(100, dtype='f8')

    # Create a vector datatype
    etype = MPI.BYTE
    filetype = etype.Create_vector(count=10, blocklength=5, stride=10)
    filetype.Commit()

    # Write with view
    fh = MPI.File.Open(comm, 'data.bin', MPI.MODE_CREATE | MPI.MODE_WRONLY)
    fh.Set_view(disp=0, etype=etype, filetype=filetype)
    fh.Write(data)
    fh.Close()
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
    def element_bytesize(self):
        """Size of element in byte units."""
        return self._numpy_type.itemsize

    @property
    def bytesize(self):
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
        if len(self._chunk_sizes) == 0:
            return 0
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
        chunk_sizes = np.tile(self._chunk_sizes, count)

        # Extent is a multiplication
        new_extent = count * extent

        return Datatype(
            self._numpy_type, positions.ravel(),
            chunk_sizes, self._lower_bound, self._lower_bound + new_extent)

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
        chunk_sizes = np.tile(self._chunk_sizes, count * blocklength)

        # Extent is a multiplication
        new_extent = count * blocklength * extent

        return Datatype(
            self._numpy_type, positions.ravel(),
            chunk_sizes, self._lower_bound, self._lower_bound + new_extent)

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
        chunk_sizes = np.tile(self._chunk_sizes, np.multiply.reduce(subsizes))

        # For subarray, the extent is the whole array
        new_extent = np.multiply.reduce(sizes) * extent

        return Datatype(
            self._numpy_type, positions.ravel(),
            chunk_sizes, self._lower_bound, self._lower_bound + new_extent)


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
    MODE_EXCL = 8
    MODE_DELETE_ON_CLOSE = 16
    MODE_UNIQUE_OPEN = 32
    MODE_SEQUENTIAL = 64
    MODE_APPEND = 128

    _MODE_6 = 6

    def __or__(self, other):
        return OpeningMode(self.value | other.value)

    def std_mode(self):
        """
        Convert MPI opening mode to Python file mode string.

        Returns
        -------
        str
            Python file mode ('rb', 'wb', 'ab', 'xb', 'r+b')

        Notes
        -----
        MODE_DELETE_ON_CLOSE, MODE_UNIQUE_OPEN, and MODE_SEQUENTIAL are
        accepted but not enforced in the stub implementation. They would
        require special handling at the File object level.
        """
        mode_bits = self.value

        # Extract access mode (lower 2 bits: RDONLY=1, WRONLY=2, RDWR=3)
        access_mode = mode_bits & 0x3

        # Check for exclusive creation
        if (self.MODE_CREATE.value & mode_bits) and (self.MODE_EXCL.value & mode_bits):
            return "xb"  # Exclusive creation, fails if file exists

        # Check for append mode
        if self.MODE_APPEND.value & mode_bits:
            return "ab"

        # Check for create mode
        if self.MODE_CREATE.value & mode_bits:
            return "wb"

        # Check access mode
        if access_mode == self.MODE_RDWR.value:
            return "r+b"
        elif access_mode == self.MODE_WRONLY.value:
            return "ab"
        else:  # MODE_RDONLY or default
            return "rb"


MODE_RDONLY = OpeningMode.MODE_RDONLY
MODE_WRONLY = OpeningMode.MODE_WRONLY
MODE_RDWR = OpeningMode.MODE_RDWR
MODE_CREATE = OpeningMode.MODE_CREATE
MODE_EXCL = OpeningMode.MODE_EXCL
MODE_DELETE_ON_CLOSE = OpeningMode.MODE_DELETE_ON_CLOSE
MODE_UNIQUE_OPEN = OpeningMode.MODE_UNIQUE_OPEN
MODE_SEQUENTIAL = OpeningMode.MODE_SEQUENTIAL
MODE_APPEND = OpeningMode.MODE_APPEND


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

    def Bcast(self, buf, root=0):
        """
        Broadcast data from root to all processes.

        In the single-process stub, this is a no-op since data is already
        present on the only process.

        Parameters
        ----------
        buf : array_like
            Data buffer. In a multi-process environment, this would be the
            data to send on the root process and would be filled on all
            other processes. In the stub, no modification occurs.
        root : int, optional
            Rank of the broadcast root (must be 0 for stub). Default is 0.

        Raises
        ------
        ValueError
            If root is not 0, since the stub only supports single-process
            execution.

        Notes
        -----
        This operation has no effect in the stub implementation because
        there is only one process. The data is already "broadcast" by
        virtue of being on the only process.

        Examples
        --------
        >>> import numpy as np
        >>> from NuMPI import MPI
        >>> data = np.array([1.0, 2.0, 3.0])
        >>> MPI.COMM_WORLD.Bcast(data, root=0)
        >>> # data remains unchanged (already on the only process)
        """
        if root != 0:
            raise ValueError("Root must be zero for MPI stub implementation.")
        # No-op: in single process, data is already "broadcast"
        pass

    bcast = Bcast  # Lowercase pickle-based version (same behavior for stub)

    def Gather(self, sendbuf, recvbuf, root=0):
        """
        Gather data from all processes to root.

        In the single-process stub, this copies sendbuf to recvbuf.

        Parameters
        ----------
        sendbuf : array_like or tuple
            Send buffer or (send buffer, send type). Data to be sent from
            this process.
        recvbuf : array_like or tuple
            Receive buffer or (receive buffer, receive type). Only used on
            the root process. In multi-process execution, this would contain
            data from all processes concatenated.
        root : int, optional
            Rank of the receiving process (must be 0 for stub). Default is 0.

        Raises
        ------
        ValueError
            If root is not 0.
        TypeError
            If send and receive buffer types do not match.

        Notes
        -----
        In the single-process stub, this simply copies sendbuf to recvbuf
        since there is only one process contributing data.

        Examples
        --------
        >>> import numpy as np
        >>> from NuMPI import MPI
        >>> sendbuf = np.array([1.0, 2.0])
        >>> recvbuf = np.zeros(2)
        >>> MPI.COMM_WORLD.Gather(sendbuf, recvbuf, root=0)
        >>> # recvbuf now contains [1.0, 2.0]
        """
        if root != 0:
            raise ValueError("Root must be zero for MPI stub implementation.")

        # Parse send buffer
        try:
            senddata = sendbuf
            sendtype = sendbuf.dtype
        except AttributeError:
            senddata, sendtype = sendbuf

        # Parse receive buffer
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

    gather = Gather  # Lowercase pickle-based version

    def Gatherv(self, sendbuf, recvbuf, root=0):
        """
        Gather data with variable-sized contributions from all processes to root.

        In the single-process stub, this behaves identically to Gather.

        Parameters
        ----------
        sendbuf : array_like or tuple
            Send buffer or (send buffer, send type).
        recvbuf : tuple
            Receive buffer specification: (recvbuf, counts, displacements, recvtype)
            or (recvbuf, counts). Only used on root.
        root : int, optional
            Rank of the receiving process (must be 0 for stub). Default is 0.

        Raises
        ------
        ValueError
            If root is not 0.
        TypeError
            If send and receive buffer types do not match.

        Notes
        -----
        The 'v' variant (variable) allows for different-sized contributions
        from each process. In the stub, this is equivalent to Gather.
        """
        if root != 0:
            raise ValueError("Root must be zero for MPI stub implementation.")

        # Parse send buffer
        try:
            senddata = sendbuf
            sendtype = sendbuf.dtype
        except AttributeError:
            senddata, sendtype = sendbuf

        # Parse receive buffer (may have counts and displacements)
        try:
            recvdata = recvbuf
            recvtype = recvbuf.dtype
        except AttributeError:
            # Could be (recvdata, counts) or (recvdata, counts, displacements, recvtype)
            if len(recvbuf) >= 2:
                recvdata = recvbuf[0]
                recvtype = recvdata.dtype if len(recvbuf) < 4 else recvbuf[3]
            else:
                raise ValueError("Invalid recvbuf format for Gatherv")

        if sendtype != recvtype:
            raise TypeError(
                "Mismatch in send and receive MPI data types in MPI stub implementation. "
                f"Send type is {sendtype} while receive type is {recvtype}."
            )

        recvdata[...] = senddata

    gatherv = Gatherv  # Lowercase pickle-based version

    def Scatter(self, sendbuf, recvbuf, root=0):
        """
        Scatter data from root to all processes.

        In the single-process stub, this copies the appropriate portion of
        sendbuf to recvbuf (simulating root sending to itself).

        Parameters
        ----------
        sendbuf : array_like or tuple
            Send buffer or (send buffer, send type). Only used on root.
            In multi-process execution, this contains data for all processes.
        recvbuf : array_like or tuple
            Receive buffer or (receive buffer, receive type). Data received
            by this process.
        root : int, optional
            Rank of the sending process (must be 0 for stub). Default is 0.

        Raises
        ------
        ValueError
            If root is not 0.
        TypeError
            If send and receive buffer types do not match.

        Notes
        -----
        In the single-process stub, this copies the portion of sendbuf that
        would be sent to rank 0 (i.e., the first recvbuf.size elements).

        Examples
        --------
        >>> import numpy as np
        >>> from NuMPI import MPI
        >>> sendbuf = np.array([1.0, 2.0, 3.0, 4.0])  # Data for 2 processes (2 each)
        >>> recvbuf = np.zeros(2)
        >>> MPI.COMM_WORLD.Scatter(sendbuf, recvbuf, root=0)
        >>> # recvbuf now contains [1.0, 2.0] (first portion)
        """
        if root != 0:
            raise ValueError("Root must be zero for MPI stub implementation.")

        # Parse send buffer
        try:
            senddata = sendbuf
            sendtype = sendbuf.dtype
        except AttributeError:
            senddata, sendtype = sendbuf

        # Parse receive buffer
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

        # Copy the portion of sendbuf that corresponds to rank 0
        recvdata[...] = senddata[:len(recvdata)]

    scatter = Scatter  # Lowercase pickle-based version

    def Scatterv(self, sendbuf, recvbuf, root=0):
        """
        Scatter data with variable-sized contributions from root to all processes.

        In the single-process stub, this behaves similarly to Scatter.

        Parameters
        ----------
        sendbuf : tuple
            Send buffer specification: (sendbuf, counts, displacements, sendtype)
            or (sendbuf, counts). Only used on root.
        recvbuf : array_like or tuple
            Receive buffer or (receive buffer, receive type).
        root : int, optional
            Rank of the sending process (must be 0 for stub). Default is 0.

        Raises
        ------
        ValueError
            If root is not 0.
        TypeError
            If send and receive buffer types do not match.

        Notes
        -----
        The 'v' variant (variable) allows for different-sized contributions
        to each process. In the stub, this extracts the portion specified
        for rank 0.
        """
        if root != 0:
            raise ValueError("Root must be zero for MPI stub implementation.")

        # Parse send buffer (may have counts and displacements)
        try:
            senddata = sendbuf
            sendtype = sendbuf.dtype
        except AttributeError:
            # Could be (senddata, counts) or (senddata, counts, displacements, sendtype)
            if len(sendbuf) >= 2:
                senddata = sendbuf[0]
                sendtype = senddata.dtype if len(sendbuf) < 4 else sendbuf[3]
            else:
                raise ValueError("Invalid sendbuf format for Scatterv")

        # Parse receive buffer
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

        # Copy the portion for rank 0
        recvdata[...] = senddata[:len(recvdata)]

    scatterv = Scatterv  # Lowercase pickle-based version

    # Unsupported operations with helpful error messages

    def Send(self, buf, dest, tag=0):
        """Point-to-point send is not implemented in the stub."""
        raise NotImplementedError(
            "Point-to-point communication (Send/Recv) is not implemented in the "
            "MPI stub. The stub only supports collective operations for single-process "
            "execution. To use Send/Recv, please install mpi4py with a real MPI "
            "implementation. Alternatively, consider using collective operations like "
            "Bcast, Gather, or Allreduce which are supported."
        )

    def Recv(self, buf=None, source=0, tag=0, status=None):
        """Point-to-point receive is not implemented in the stub."""
        raise NotImplementedError(
            "Point-to-point communication (Send/Recv) is not implemented in the "
            "MPI stub. The stub only supports collective operations for single-process "
            "execution. To use Send/Recv, please install mpi4py with a real MPI "
            "implementation. See Send() documentation for details."
        )

    def Isend(self, buf, dest, tag=0):
        """Non-blocking send is not implemented in the stub."""
        raise NotImplementedError(
            "Non-blocking communication (Isend/Irecv) is not implemented in the "
            "MPI stub. To use non-blocking operations and Request objects, please "
            "install mpi4py with a real MPI implementation."
        )

    def Irecv(self, buf, source=0, tag=0):
        """Non-blocking receive is not implemented in the stub."""
        raise NotImplementedError(
            "Non-blocking communication (Isend/Irecv) is not implemented in the "
            "MPI stub. To use non-blocking operations and Request objects, please "
            "install mpi4py with a real MPI implementation."
        )

    def Sendrecv(self, sendbuf, dest, sendtag=0, recvbuf=None, source=0, recvtag=0, status=None):
        """Send and receive is not implemented in the stub."""
        raise NotImplementedError(
            "Point-to-point communication (Sendrecv) is not implemented in the "
            "MPI stub. Please install mpi4py for point-to-point operations."
        )

    def Probe(self, source=0, tag=0, status=None):
        """Message probing is not implemented in the stub."""
        raise NotImplementedError(
            "Message probing (Probe/Iprobe) is not implemented in the MPI stub. "
            "Please install mpi4py for point-to-point communication features."
        )

    def Iprobe(self, source=0, tag=0, status=None):
        """Non-blocking message probing is not implemented in the stub."""
        raise NotImplementedError(
            "Message probing (Probe/Iprobe) is not implemented in the MPI stub. "
            "Please install mpi4py for point-to-point communication features."
        )

    # Lowercase versions for unsupported operations
    send = Send
    recv = Recv
    isend = Isend
    irecv = Irecv
    sendrecv = Sendrecv
    probe = Probe
    iprobe = Iprobe


# ## Stub file I/O object


class File(object):
    @classmethod
    def Open(cls, comm, filename, amode=MODE_RDONLY, info=None):
        """
        Open a file for MPI I/O.

        Parameters
        ----------
        comm : Intracomm
            MPI communicator
        filename : str or file-like
            Path to file or file-like object
        amode : OpeningMode, optional
            File opening mode (default: MODE_RDONLY)
        info : None, optional
            MPI Info object for setting file access hints. Accepted but ignored
            in the stub implementation, as it's used for optimization hints in
            real MPI implementations.

        Returns
        -------
        File
            Opened MPI file object
        """
        # info parameter is accepted but ignored in stub implementation
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
        return (self._file.tell() - self._view_start) // self._etype.bytesize

    def Read(self, buf):
        try:
            if self._filetype is None:
                self._file.seek(self._view_start)
                # read until the buffer is full
                nb_bytes = buf.size * buf.itemsize
                data = self._file.read(nb_bytes)
            else:
                e_size = self._filetype.element_bytesize
                data = bytearray()
                for position, size in self._filetype.iterate_chunks():
                    self._file.seek(self._view_start + position * e_size)
                    data += self._file.read(size * e_size)
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
            e_size = self._filetype.element_bytesize
            i_byte = 0
            for position, size in self._filetype.iterate_chunks():
                self._file.seek(self._view_start + position * e_size)
                nb_bytes = size * e_size
                self._file.write(buf[i_byte: i_byte + nb_bytes])
                i_byte += nb_bytes
        self._view_start = self._file.tell()

    Write_all = Write


COMM_WORLD = Intracomm()
COMM_SELF = Intracomm()
