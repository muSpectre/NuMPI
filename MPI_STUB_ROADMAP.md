# MPI Stub Implementation Roadmap

**Document Version:** 1.0
**Date:** 2026-01-04
**Status:** Draft

## Executive Summary

The NuMPI MPI stub is **functionally complete for current project needs** but has gaps in testing, documentation, and future-proofing. This roadmap prioritizes actionable improvements.

**Key Finding:** No missing MPI methods are blocking current development. Focus should be on **quality, maintainability, and future-proofing**.

---

## Priority 1: Critical Quality Improvements (Do First)

### 1.1 Add Direct Unit Tests for MPIStub

**Why:** Currently, MPIStub is only tested indirectly through integration tests. Direct unit tests would:
- Catch regressions when modifying the stub
- Document expected behavior clearly
- Ensure stub behavior matches mpi4py for single-process case
- Make it safe to refactor

**Action Items:**
- [ ] Create `test/test_mpi_stub.py`
- [ ] Test `Intracomm` methods: `Reduce()`, `Allreduce()`, `Allgather()`, `Allgatherv()`
- [ ] Test `Datatype` constructors: `Create_contiguous()`, `Create_vector()`, `Create_subarray()`
- [ ] Test `File` operations with various modes
- [ ] Test error conditions and edge cases
- [ ] Test that stub raises errors for root != 0 in operations

**Estimated Effort:** 4-6 hours
**Risk:** Low
**Impact:** High - improves maintainability

**Example Test Structure:**
```python
# test/test_mpi_stub.py
import pytest
import numpy as np
from NuMPI import MPIStub

def test_reduce_copies_data():
    """Test that Reduce correctly copies sendbuf to recvbuf."""
    comm = MPIStub.Intracomm()
    sendbuf = np.array([1.0, 2.0, 3.0])
    recvbuf = np.zeros(3)
    comm.Reduce(sendbuf, recvbuf)
    np.testing.assert_array_equal(recvbuf, sendbuf)

def test_reduce_rejects_nonzero_root():
    """Test that Reduce raises ValueError for root != 0."""
    comm = MPIStub.Intracomm()
    sendbuf = np.array([1.0])
    recvbuf = np.zeros(1)
    with pytest.raises(ValueError, match="Root must be zero"):
        comm.Reduce(sendbuf, recvbuf, root=1)

def test_datatype_create_contiguous():
    """Test Create_contiguous datatype constructor."""
    dtype = MPIStub.BasicDatatype('f8')
    new_dtype = dtype.Create_contiguous(3)
    assert new_dtype.Get_size() == 3 * 8  # 3 doubles

# ... more tests
```

---

### 1.2 Add Comprehensive API Documentation

**Why:** The module docstring (line 26-33) says it's incomplete but doesn't specify what's supported. Users need clarity.

**Action Items:**
- [ ] Document all supported methods in module-level docstring
- [ ] Add "Supported Operations" section listing what works
- [ ] Add "Unsupported Operations" section listing what's missing
- [ ] Add usage examples showing stub vs mpi4py patterns
- [ ] Document semantic differences from real MPI (e.g., root=0 requirement)

**Estimated Effort:** 2-3 hours
**Risk:** Low
**Impact:** High - improves usability

**Proposed Documentation Addition:**
```python
"""
Stub implementation of mpi4py. This is necessary to run a serial version of
NuMPI and dependent projects without an MPI installation.

Important note: This is at present *not* the complete API, but only includes
the features that we need in our projects. If you miss some functionality,
implement it and submit it back to us.

## Supported MPI Operations

### Communicator Methods (Intracomm)
- `Barrier()` / `barrier` - No-op for single process
- `Get_rank()` / `rank` - Always returns 0
- `Get_size()` / `size` - Always returns 1
- `Reduce(sendbuf, recvbuf, op, root)` - Copies sendbuf to recvbuf (root must be 0)
- `Allreduce(sendbuf, recvbuf, op)` - Alias for Reduce
- `Allgather(sendbuf, recvbuf)` - Copies sendbuf to recvbuf
- `Allgatherv(sendbuf, recvbuf)` - Alias for Allgather

### File I/O (File)
- `Open(comm, filename, amode)` - Open file for MPI I/O
- `Close()` - Close file
- `Read(buf)` / `Read_all(buf)` - Read data
- `Write(buf)` / `Write_all(buf)` - Write data
- `Set_view(disp, etype, filetype)` - Set file view
- `Get_position()` - Get file pointer position

### Datatypes (Datatype)
- `Create_contiguous(count)` - Create contiguous datatype
- `Create_vector(count, blocklength, stride)` - Create vector datatype
- `Create_subarray(sizes, subsizes, starts, order)` - Create subarray datatype
- `Commit()` - Commit datatype (no-op)
- `Free()` - Free datatype (no-op)
- `Get_size()` - Get datatype size

### Constants
- Operations: MIN, MAX, SUM, PROD, LAND, BAND, LOR, BOR, LXOR, BXOR, MAXLOC, MINLOC
- File modes: MODE_RDONLY, MODE_WRONLY, MODE_RDWR, MODE_CREATE
- Layouts: ORDER_C, ORDER_F
- Communicators: COMM_WORLD, COMM_SELF

## Unsupported Operations

The following MPI operations are **NOT** implemented:

### Point-to-Point Communication
- Send, Recv, Isend, Irecv, Sendrecv, Probe, Iprobe

### Collective Operations
- Bcast, Scatter, Scatterv, Gather, Gatherv
- Alltoall, Alltoallv, Alltoallw
- Reduce_scatter, Scan, Exscan

### Advanced Features
- Request objects and non-blocking operation management
- Communicator management (Dup, Split, Create, Free)
- Group operations
- Topology (Cartesian, Graph)

If you need these features, please install mpi4py or contribute implementations.

## Usage Notes

1. **Root Rank Restriction:** All collective operations that take a `root` parameter
   only accept `root=0` in the stub, since there's only one process.

2. **Type Safety:** The stub checks that send and receive buffers have matching dtypes.

3. **File I/O:** File I/O operations support both file paths and file-like objects
   (not supported when using real mpi4py).

## Example Usage

```python
from NuMPI import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()  # Always 0 with stub
size = comm.Get_size()  # Always 1 with stub

# This works with both stub and mpi4py
sendbuf = np.array([1.0, 2.0, 3.0])
recvbuf = np.zeros(3)
comm.Allreduce(sendbuf, recvbuf, op=MPI.SUM)
```
"""
```

---

## Priority 2: Fix Existing FIXMEs (Quick Wins)

### 2.1 Implement Missing File Opening Modes

**Current Issue:** Lines 305-336 mark several file modes as unsupported:
- MODE_EXCL
- MODE_DELETE_ON_CLOSE
- MODE_UNIQUE_OPEN
- MODE_SEQUENTIAL
- MODE_APPEND

**Why Fix:** Low-hanging fruit, improves MPI standard conformance

**Action Items:**
- [ ] Research these modes in MPI-3.1 standard
- [ ] Implement mode behavior in `OpeningMode.std_mode()` method
- [ ] Add tests for each mode
- [ ] Remove FIXME comments

**Estimated Effort:** 2-3 hours
**Risk:** Low
**Impact:** Medium - improves completeness

**Implementation Sketch:**
```python
class OpeningMode(Enum):
    MODE_RDONLY = 1
    MODE_WRONLY = 2
    MODE_RDWR = 3
    MODE_CREATE = 4
    MODE_EXCL = 8       # Fail if file exists (with CREATE)
    MODE_DELETE_ON_CLOSE = 16
    MODE_UNIQUE_OPEN = 32
    MODE_SEQUENTIAL = 64
    MODE_APPEND = 128

    def std_mode(self):
        """Convert MPI mode to Python file mode."""
        mode_bits = self.value

        # Check for incompatible combinations
        if (mode_bits & self.MODE_RDONLY.value) and (mode_bits & self.MODE_WRONLY.value):
            raise ValueError("Cannot combine MODE_RDONLY and MODE_WRONLY")

        # Determine base mode
        if mode_bits & self.MODE_CREATE.value:
            if mode_bits & self.MODE_EXCL.value:
                base = "xb"  # Exclusive creation
            else:
                base = "wb"
        elif mode_bits & self.MODE_APPEND.value:
            base = "ab"
        elif mode_bits & self.MODE_WRONLY.value:
            base = "ab"  # Write-only opens in append mode
        elif mode_bits & self.MODE_RDWR.value:
            base = "r+b"
        else:
            base = "rb"

        # Note: MODE_DELETE_ON_CLOSE, MODE_UNIQUE_OPEN, MODE_SEQUENTIAL
        # would need special handling in File.__init__ and Close()
        return base
```

---

### 2.2 Add `info` Parameter to File.Open()

**Current Issue:** Line 414 notes the `info` parameter is missing

**Why Fix:** Improves MPI standard conformance, though info is rarely used in practice

**Action Items:**
- [ ] Add optional `info` parameter to `File.Open()` classmethod
- [ ] Add optional `info` parameter to `File.__init__()`
- [ ] Document that info is accepted but ignored (appropriate for stub)
- [ ] Remove FIXME comment

**Estimated Effort:** 30 minutes
**Risk:** Very Low
**Impact:** Low - rarely used parameter

**Implementation:**
```python
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
    amode : OpeningMode
        File opening mode
    info : None, optional
        MPI Info object (accepted but ignored in stub implementation)

    Returns
    -------
    File
        Opened MPI file object
    """
    # info parameter is accepted but ignored in stub
    return File(comm, filename, amode)
```

---

## Priority 3: Future-Proofing (Prepare for Growth)

### 3.1 Implement Commonly-Needed Collective Operations

**Why:** While not currently used, these are very common in MPI codes. Adding them now prevents future breakage.

**Most Valuable Additions (in order):**

#### 3.1.1 Bcast() - Broadcast

**Impact:** HIGH - One of the most common MPI operations
**Effort:** 30 minutes

```python
def Bcast(self, buf, root=0):
    """
    Broadcast data from root to all processes.

    In single-process stub, this is a no-op since data is already
    where it needs to be.

    Parameters
    ----------
    buf : array_like
        Data buffer (modified in-place on non-root processes)
    root : int
        Rank of broadcast root (must be 0 for stub)
    """
    if root != 0:
        raise ValueError("Root must be zero for MPI stub implementation.")
    # No-op: in single process, data is already "broadcast"
    pass

bcast = Bcast  # Lowercase pickle-based version (same for stub)
```

**Tests needed:**
```python
def test_bcast_is_noop():
    comm = MPIStub.Intracomm()
    buf = np.array([1.0, 2.0, 3.0])
    original = buf.copy()
    comm.Bcast(buf, root=0)
    np.testing.assert_array_equal(buf, original)

def test_bcast_rejects_nonzero_root():
    comm = MPIStub.Intracomm()
    buf = np.array([1.0])
    with pytest.raises(ValueError):
        comm.Bcast(buf, root=1)
```

---

#### 3.1.2 Gather() / Gatherv() - Gather to Root

**Impact:** MEDIUM - Complements existing Allgather
**Effort:** 1 hour

```python
def Gather(self, sendbuf, recvbuf, root=0):
    """
    Gather data from all processes to root.

    In single-process stub, this copies sendbuf to recvbuf.

    Parameters
    ----------
    sendbuf : array_like or tuple
        Send buffer or (send buffer, send type)
    recvbuf : array_like or tuple
        Receive buffer or (receive buffer, receive type) - only used on root
    root : int
        Rank of receiving process (must be 0 for stub)
    """
    if root != 0:
        raise ValueError("Root must be zero for MPI stub implementation.")

    # Parse buffers (same logic as Allgather)
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
            f"Mismatch in send and receive MPI data types. "
            f"Send type is {sendtype} while receive type is {recvtype}."
        )

    recvdata[...] = senddata

gather = Gather  # Lowercase version
Gatherv = Gather  # Vector version (same for stub)
gatherv = Gather  # Lowercase vector version
```

---

#### 3.1.3 Scatter() / Scatterv() - Scatter from Root

**Impact:** MEDIUM - Useful for distributing work
**Effort:** 1 hour

```python
def Scatter(self, sendbuf, recvbuf, root=0):
    """
    Scatter data from root to all processes.

    In single-process stub, this copies the first portion of sendbuf
    to recvbuf (simulating root sending to itself).

    Parameters
    ----------
    sendbuf : array_like or tuple
        Send buffer on root (ignored on non-root)
    recvbuf : array_like or tuple
        Receive buffer
    root : int
        Rank of sending process (must be 0 for stub)
    """
    if root != 0:
        raise ValueError("Root must be zero for MPI stub implementation.")

    # Parse buffers
    try:
        senddata = sendbuf
    except AttributeError:
        senddata, _ = sendbuf

    try:
        recvdata = recvbuf
    except AttributeError:
        recvdata, _ = recvbuf

    # Copy portion of sendbuf that would go to rank 0
    recvdata[...] = senddata[:len(recvdata)]

scatter = Scatter
Scatterv = Scatter
scatterv = Scatter
```

---

### 3.2 Add Helpful Error Messages for Unsupported Methods

**Why:** When users try to use unsupported features, they should get clear guidance

**Action Items:**
- [ ] Add stub methods for common unsupported operations
- [ ] Each raises `NotImplementedError` with helpful message
- [ ] Message should suggest installing mpi4py or contributing

**Estimated Effort:** 1 hour
**Risk:** Low
**Impact:** Medium - improves user experience

**Implementation:**
```python
class Intracomm(object):
    # ... existing methods ...

    def Send(self, buf, dest, tag=0):
        """Point-to-point send is not implemented in the stub."""
        raise NotImplementedError(
            "Point-to-point communication (Send/Recv) is not implemented in the "
            "MPI stub. The stub only supports collective operations for single-process "
            "execution. To use Send/Recv, please install mpi4py with a real MPI "
            "implementation. Alternatively, consider using collective operations like "
            "Bcast, Gather, or Allreduce which are supported."
        )

    def Recv(self, buf, source, tag=0, status=None):
        """Point-to-point receive is not implemented in the stub."""
        raise NotImplementedError(
            "Point-to-point communication (Send/Recv) is not implemented in the "
            "MPI stub. See Send() for details."
        )

    send = Send
    recv = Recv

    def Isend(self, buf, dest, tag=0):
        """Non-blocking send is not implemented in the stub."""
        raise NotImplementedError(
            "Non-blocking communication is not implemented in the MPI stub. "
            "To use Isend/Irecv/Request objects, please install mpi4py."
        )

    Irecv = Isend  # Same error message
    isend = Isend
    irecv = Isend
```

---

## Priority 4: Long-Term Enhancements (Future Work)

### 4.1 Create Comprehensive Comparison Tests

**Why:** Ensure stub behavior exactly matches mpi4py in single-process case

**Action Items:**
- [ ] Create `test/test_stub_vs_mpi4py.py`
- [ ] Mark tests as `@pytest.mark.skipif(not _has_mpi4py)`
- [ ] Run same operations with both MPI and MPIStub
- [ ] Assert results are identical

**Estimated Effort:** 4-6 hours
**Risk:** Low
**Impact:** High - ensures compatibility

---

### 4.2 Add Type Hints Throughout

**Why:** Modern Python best practice, helps with IDE support and static analysis

**Action Items:**
- [ ] Add type hints to all method signatures
- [ ] Add type hints to all return values
- [ ] Use `typing` module for complex types
- [ ] Run mypy to validate

**Estimated Effort:** 3-4 hours
**Risk:** Low
**Impact:** Medium - improves maintainability

**Example:**
```python
from typing import Optional, Union, Tuple, Any
import numpy.typing as npt

class Intracomm:
    def Barrier(self) -> None:
        pass

    def Get_rank(self) -> int:
        return 0

    def Reduce(
        self,
        sendbuf: Union[npt.NDArray[Any], Tuple[npt.NDArray[Any], Any]],
        recvbuf: Union[npt.NDArray[Any], Tuple[npt.NDArray[Any], Any]],
        op: Operations = Operations.SUM,
        root: int = 0
    ) -> None:
        # ...
```

---

### 4.3 Performance Optimization

**Why:** While correctness is primary, efficiency matters for large-scale I/O

**Action Items:**
- [ ] Profile `Datatype.merge_contiguous_chunks()` (line 124-151)
- [ ] Profile `File.Read()` and `File.Write()` with large arrays
- [ ] Consider caching in Datatype operations
- [ ] Benchmark against numpy direct I/O

**Estimated Effort:** 6-8 hours
**Risk:** Medium - could introduce bugs
**Impact:** Low to Medium - depends on usage patterns

---

## Implementation Timeline Recommendation

### Sprint 1 (1-2 days)
- ✅ Priority 1.1: Add unit tests for MPIStub
- ✅ Priority 1.2: Add comprehensive API documentation
- **Deliverable:** Well-tested, well-documented stub

### Sprint 2 (Half day)
- ✅ Priority 2.1: Implement missing file modes
- ✅ Priority 2.2: Add info parameter
- **Deliverable:** All FIXMEs resolved

### Sprint 3 (1 day)
- ✅ Priority 3.1.1: Implement Bcast
- ✅ Priority 3.1.2: Implement Gather/Gatherv
- ✅ Priority 3.1.3: Implement Scatter/Scatterv
- ✅ Priority 3.2: Add error messages for unsupported methods
- **Deliverable:** More complete MPI API coverage

### Sprint 4+ (Future work)
- Priority 4.1: Comparison tests with mpi4py
- Priority 4.2: Add type hints
- Priority 4.3: Performance optimization

---

## Success Metrics

- **Test Coverage:** Achieve >90% line coverage for MPIStub.py
- **Documentation:** Zero TODOs/FIXMEs remaining in MPIStub.py
- **User Experience:** Zero issues filed about unclear stub behavior
- **Compatibility:** All stub methods behave identically to mpi4py in single-process case

---

## Non-Goals

**What NOT to implement:**

1. **Full MPI Standard Compliance** - This is a stub for serial execution, not a complete MPI implementation
2. **Multi-Process Support** - The stub is explicitly for single-process use
3. **Performance Parity with MPI** - Correctness over speed
4. **Derived Datatypes Beyond Current Support** - Only add if needed

---

## Questions for Maintainers

1. **Testing Strategy:** Should we run the test suite with both mpi4py and stub to ensure compatibility?
2. **Version Targeting:** Which MPI standard version should we target? (Currently references MPI 5.0)
3. **Feature Requests:** Are there specific MPI features you'd like to use but currently can't?
4. **Breaking Changes:** Is it acceptable to require Python 3.8+ for type hints?

---

## Appendix: Quick Reference

### What Works Today
✅ Barrier, Get_rank, Get_size
✅ Reduce, Allreduce
✅ Allgather, Allgatherv
✅ File I/O (Open, Read, Write, Close, Set_view)
✅ Datatypes (Create_contiguous, Create_vector, Create_subarray)

### What's Missing (Not Used in Project)
❌ Point-to-point (Send, Recv, Isend, Irecv)
❌ Bcast, Scatter, Gather (non-All versions)
❌ Alltoall, Reduce_scatter, Scan
❌ Communicator management
❌ Groups, Topologies, Requests

### What Should Be Added Next
🎯 Priority 1: Unit tests + documentation
🎯 Priority 2: Fix FIXMEs (file modes, info parameter)
🎯 Priority 3: Bcast, Gather, Scatter + error messages
