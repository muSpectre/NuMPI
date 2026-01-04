#
# Copyright 2026 Claude AI Assistant
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
Unit tests for MPIStub module.

These tests verify that the MPI stub implementation behaves correctly
for single-process serial execution.
"""

import os
import tempfile
import pytest
import numpy as np

from NuMPI import MPIStub


# =============================================================================
# Intracomm Tests
# =============================================================================


class TestIntracomm:
    """Tests for the Intracomm communicator class."""

    def test_get_rank(self):
        """Test that Get_rank returns 0."""
        comm = MPIStub.Intracomm()
        assert comm.Get_rank() == 0

    def test_rank_property(self):
        """Test that rank property returns 0."""
        comm = MPIStub.Intracomm()
        assert comm.rank == 0

    def test_get_size(self):
        """Test that Get_size returns 1."""
        comm = MPIStub.Intracomm()
        assert comm.Get_size() == 1

    def test_size_property(self):
        """Test that size property returns 1."""
        comm = MPIStub.Intracomm()
        assert comm.size == 1

    def test_barrier(self):
        """Test that Barrier executes without error (no-op)."""
        comm = MPIStub.Intracomm()
        comm.Barrier()  # Should not raise

    def test_barrier_lowercase(self):
        """Test that lowercase barrier executes without error."""
        comm = MPIStub.Intracomm()
        comm.barrier()  # Should not raise

    def test_reduce_with_arrays(self):
        """Test Reduce with numpy arrays."""
        comm = MPIStub.Intracomm()
        sendbuf = np.array([1.0, 2.0, 3.0])
        recvbuf = np.zeros(3)
        comm.Reduce(sendbuf, recvbuf, op=MPIStub.SUM, root=0)
        np.testing.assert_array_equal(recvbuf, sendbuf)

    def test_reduce_with_tuples(self):
        """Test Reduce with (buffer, type) tuples."""
        comm = MPIStub.Intracomm()
        sendbuf = np.array([1.0, 2.0, 3.0])
        recvbuf = np.zeros(3)
        comm.Reduce(
            (sendbuf, sendbuf.dtype),
            (recvbuf, recvbuf.dtype),
            op=MPIStub.SUM,
            root=0
        )
        np.testing.assert_array_equal(recvbuf, sendbuf)

    def test_reduce_rejects_nonzero_root(self):
        """Test that Reduce raises ValueError for root != 0."""
        comm = MPIStub.Intracomm()
        sendbuf = np.array([1.0])
        recvbuf = np.zeros(1)
        with pytest.raises(ValueError, match="Root must be zero"):
            comm.Reduce(sendbuf, recvbuf, root=1)

    def test_reduce_rejects_type_mismatch(self):
        """Test that Reduce raises TypeError for mismatched types."""
        comm = MPIStub.Intracomm()
        sendbuf = np.array([1.0], dtype=np.float64)
        recvbuf = np.zeros(1, dtype=np.int32)
        with pytest.raises(TypeError, match="Mismatch in send and receive"):
            comm.Reduce(sendbuf, recvbuf)

    def test_allreduce(self):
        """Test Allreduce with arrays."""
        comm = MPIStub.Intracomm()
        sendbuf = np.array([1.0, 2.0, 3.0])
        recvbuf = np.zeros(3)
        comm.Allreduce(sendbuf, recvbuf, op=MPIStub.MAX)
        np.testing.assert_array_equal(recvbuf, sendbuf)

    def test_allreduce_is_reduce(self):
        """Test that Allreduce is aliased to Reduce."""
        assert MPIStub.Intracomm.Allreduce == MPIStub.Intracomm.Reduce

    def test_allgather_with_arrays(self):
        """Test Allgather with numpy arrays."""
        comm = MPIStub.Intracomm()
        sendbuf = np.array([1.0, 2.0, 3.0])
        recvbuf = np.zeros(3)
        comm.Allgather(sendbuf, (recvbuf, None))
        np.testing.assert_array_equal(recvbuf, sendbuf)

    def test_allgather_rejects_type_mismatch(self):
        """Test that Allgather raises TypeError for mismatched types."""
        comm = MPIStub.Intracomm()
        sendbuf = np.array([1.0], dtype=np.float64)
        recvbuf = np.zeros(1, dtype=np.int32)
        with pytest.raises(TypeError, match="Mismatch in send and receive"):
            comm.Allgather(sendbuf, (recvbuf, None))

    def test_allgatherv_is_allgather(self):
        """Test that Allgatherv is aliased to Allgather."""
        assert MPIStub.Intracomm.Allgatherv == MPIStub.Intracomm.Allgather


# =============================================================================
# Datatype Tests
# =============================================================================


class TestDatatype:
    """Tests for the Datatype class."""

    def test_basic_datatype_creation(self):
        """Test creating a basic datatype."""
        dtype = MPIStub.BasicDatatype('f8')
        assert dtype.element_bytesize == 8

    def test_datatype_get_size(self):
        """Test Datatype.Get_size()."""
        dtype = MPIStub.BasicDatatype('f8')
        assert dtype.Get_size() == 8

    def test_datatype_commit(self):
        """Test Datatype.Commit() (no-op)."""
        dtype = MPIStub.BasicDatatype('f8')
        dtype.Commit()  # Should not raise

    def test_datatype_free(self):
        """Test Datatype.Free() (no-op)."""
        dtype = MPIStub.BasicDatatype('f8')
        dtype.Free()  # Should not raise

    def test_datatype_free_lowercase(self):
        """Test Datatype.free() (no-op)."""
        dtype = MPIStub.BasicDatatype('f8')
        dtype.free()  # Should not raise

    def test_create_contiguous(self):
        """Test Create_contiguous with count > 0."""
        dtype = MPIStub.BasicDatatype('f8')
        new_dtype = dtype.Create_contiguous(3)
        assert new_dtype.Get_size() == 3 * 8  # 3 doubles

    def test_create_contiguous_zero(self):
        """Test Create_contiguous with count = 0."""
        dtype = MPIStub.BasicDatatype('f8')
        new_dtype = dtype.Create_contiguous(0)
        assert new_dtype.Get_size() == 0

    def test_create_vector(self):
        """Test Create_vector."""
        dtype = MPIStub.BasicDatatype('f8')
        new_dtype = dtype.Create_vector(count=2, blocklength=3, stride=4)
        # 2 blocks * 3 elements * 8 bytes = 48 bytes
        assert new_dtype.Get_size() == 2 * 3 * 8

    def test_create_vector_zero_count(self):
        """Test Create_vector with count = 0."""
        dtype = MPIStub.BasicDatatype('f8')
        new_dtype = dtype.Create_vector(count=0, blocklength=3, stride=4)
        assert new_dtype.Get_size() == 0

    def test_create_vector_zero_blocklength(self):
        """Test Create_vector with blocklength = 0."""
        dtype = MPIStub.BasicDatatype('f8')
        new_dtype = dtype.Create_vector(count=2, blocklength=0, stride=4)
        assert new_dtype.Get_size() == 0

    def test_create_subarray(self):
        """Test Create_subarray with C order."""
        dtype = MPIStub.BasicDatatype('f8')
        sizes = [10, 20]
        subsizes = [5, 10]
        starts = [0, 0]
        new_dtype = dtype.Create_subarray(sizes, subsizes, starts, MPIStub.ORDER_C)
        # 5 * 10 elements * 8 bytes = 400 bytes
        assert new_dtype.Get_size() == 5 * 10 * 8

    def test_create_subarray_fortran_order(self):
        """Test Create_subarray with Fortran order."""
        dtype = MPIStub.BasicDatatype('f8')
        sizes = [10, 20]
        subsizes = [5, 10]
        starts = [0, 0]
        new_dtype = dtype.Create_subarray(sizes, subsizes, starts, MPIStub.ORDER_F)
        assert new_dtype.Get_size() == 5 * 10 * 8

    def test_create_subarray_zero_subsize(self):
        """Test Create_subarray with zero subsize."""
        dtype = MPIStub.BasicDatatype('f8')
        sizes = [10, 20]
        subsizes = [0, 10]
        starts = [0, 0]
        new_dtype = dtype.Create_subarray(sizes, subsizes, starts, MPIStub.ORDER_C)
        assert new_dtype.Get_size() == 0

    def test_merge_contiguous_chunks(self):
        """Test merge_contiguous_chunks static method."""
        positions = [0, 1, 2, 5, 6, 10]
        sizes = [1, 1, 1, 1, 1, 2]
        merged_pos, merged_sizes = MPIStub.Datatype.merge_contiguous_chunks(positions, sizes)
        # Should merge [0,1,2] and [5,6]
        expected_pos = [0, 5, 10]
        expected_sizes = [3, 2, 2]
        np.testing.assert_array_equal(merged_pos, expected_pos)
        np.testing.assert_array_equal(merged_sizes, expected_sizes)

    def test_merge_contiguous_chunks_empty(self):
        """Test merge_contiguous_chunks with empty input."""
        positions = []
        sizes = []
        merged_pos, merged_sizes = MPIStub.Datatype.merge_contiguous_chunks(positions, sizes)
        assert len(merged_pos) == 0
        assert len(merged_sizes) == 0


# =============================================================================
# Operations Tests
# =============================================================================


class TestOperations:
    """Tests for MPI operations enums."""

    def test_operations_exist(self):
        """Test that all operations are defined."""
        assert MPIStub.MIN is not None
        assert MPIStub.MAX is not None
        assert MPIStub.SUM is not None
        assert MPIStub.PROD is not None
        assert MPIStub.LAND is not None
        assert MPIStub.BAND is not None
        assert MPIStub.LOR is not None
        assert MPIStub.BOR is not None
        assert MPIStub.LXOR is not None
        assert MPIStub.BXOR is not None
        assert MPIStub.MAXLOC is not None
        assert MPIStub.MINLOC is not None


# =============================================================================
# File I/O Tests
# =============================================================================


class TestFile:
    """Tests for the File class."""

    def test_file_open_and_close(self):
        """Test opening and closing a file."""
        comm = MPIStub.Intracomm()
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp_name = tmp.name

        try:
            f = MPIStub.File.Open(comm, tmp_name, MPIStub.MODE_CREATE | MPIStub.MODE_WRONLY)
            f.Close()
        finally:
            if os.path.exists(tmp_name):
                os.remove(tmp_name)

    def test_file_write_and_read(self):
        """Test writing and reading data."""
        comm = MPIStub.Intracomm()
        data = np.array([1.0, 2.0, 3.0, 4.0])

        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp_name = tmp.name

        try:
            # Write
            f = MPIStub.File.Open(comm, tmp_name, MPIStub.MODE_CREATE | MPIStub.MODE_WRONLY)
            f.Write(data)
            f.Close()

            # Read
            f = MPIStub.File.Open(comm, tmp_name, MPIStub.MODE_RDONLY)
            read_data = np.zeros(4)
            f.Read(read_data)
            f.Close()

            np.testing.assert_array_equal(read_data, data)
        finally:
            if os.path.exists(tmp_name):
                os.remove(tmp_name)

    def test_file_set_view(self):
        """Test Set_view method."""
        comm = MPIStub.Intracomm()
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp_name = tmp.name

        try:
            f = MPIStub.File.Open(comm, tmp_name, MPIStub.MODE_CREATE | MPIStub.MODE_WRONLY)
            etype = MPIStub.BasicDatatype('f8')
            f.Set_view(disp=0, etype=etype, filetype=None)
            f.Close()
        finally:
            if os.path.exists(tmp_name):
                os.remove(tmp_name)

    def test_file_get_position(self):
        """Test Get_position method."""
        comm = MPIStub.Intracomm()
        data = np.array([1.0, 2.0, 3.0])

        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp_name = tmp.name

        try:
            f = MPIStub.File.Open(comm, tmp_name, MPIStub.MODE_CREATE | MPIStub.MODE_WRONLY)
            f.Write(data)
            pos = f.Get_position()
            f.Close()
            assert pos > 0
        finally:
            if os.path.exists(tmp_name):
                os.remove(tmp_name)

    def test_file_read_all_is_read(self):
        """Test that Read_all is aliased to Read."""
        assert MPIStub.File.Read_all == MPIStub.File.Read

    def test_file_write_all_is_write(self):
        """Test that Write_all is aliased to Write."""
        assert MPIStub.File.Write_all == MPIStub.File.Write

    def test_file_requires_intracomm(self):
        """Test that File requires Intracomm communicator."""
        with pytest.raises(RuntimeError, match="Communicator object must be an instance"):
            MPIStub.File(None, "dummy.txt", MPIStub.MODE_RDONLY)


# =============================================================================
# Opening Mode Tests
# =============================================================================


class TestOpeningMode:
    """Tests for the OpeningMode enum."""

    def test_mode_rdonly(self):
        """Test MODE_RDONLY."""
        assert MPIStub.MODE_RDONLY.std_mode() == "rb"

    def test_mode_wronly(self):
        """Test MODE_WRONLY."""
        assert MPIStub.MODE_WRONLY.std_mode() == "ab"

    def test_mode_create(self):
        """Test MODE_CREATE."""
        assert MPIStub.MODE_CREATE.std_mode() == "wb"

    def test_mode_create_wronly(self):
        """Test MODE_CREATE | MODE_WRONLY."""
        mode = MPIStub.MODE_CREATE | MPIStub.MODE_WRONLY
        assert mode.std_mode() == "wb"

    def test_mode_bitwise_or(self):
        """Test that modes can be combined with | operator."""
        mode = MPIStub.MODE_CREATE | MPIStub.MODE_WRONLY
        assert isinstance(mode, MPIStub.OpeningMode)


# =============================================================================
# Layout Tests
# =============================================================================


class TestLayout:
    """Tests for the Layout enum."""

    def test_order_c_exists(self):
        """Test that ORDER_C is defined."""
        assert MPIStub.ORDER_C is not None
        assert MPIStub.Layout.ORDER_C is not None

    def test_order_f_exists(self):
        """Test that ORDER_F is defined."""
        assert MPIStub.ORDER_F is not None
        assert MPIStub.Layout.ORDER_F is not None


# =============================================================================
# Communicator Constants Tests
# =============================================================================


class TestCommunicatorConstants:
    """Tests for predefined communicators."""

    def test_comm_world_exists(self):
        """Test that COMM_WORLD is defined."""
        assert MPIStub.COMM_WORLD is not None
        assert isinstance(MPIStub.COMM_WORLD, MPIStub.Intracomm)

    def test_comm_self_exists(self):
        """Test that COMM_SELF is defined."""
        assert MPIStub.COMM_SELF is not None
        assert isinstance(MPIStub.COMM_SELF, MPIStub.Intracomm)

    def test_comm_world_properties(self):
        """Test COMM_WORLD properties."""
        assert MPIStub.COMM_WORLD.rank == 0
        assert MPIStub.COMM_WORLD.size == 1

    def test_comm_self_properties(self):
        """Test COMM_SELF properties."""
        assert MPIStub.COMM_SELF.rank == 0
        assert MPIStub.COMM_SELF.size == 1


# =============================================================================
# Typedict Tests
# =============================================================================


class TestTypedict:
    """Tests for the Typedict class."""

    def test_byte_type(self):
        """Test BYTE datatype."""
        assert MPIStub.BYTE is not None
        assert MPIStub.BYTE.element_bytesize == 1

    def test_typedict_access(self):
        """Test accessing types through typedict."""
        typedict = MPIStub._typedict
        dtype = typedict["f8"]
        assert dtype.element_bytesize == 8
