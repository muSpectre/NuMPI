# MPI Stub: Actionable Next Steps

**TL;DR:** The stub works well today. Focus on **testing and documentation** first, then add commonly-needed methods.

---

## Immediate Actions (Do This Week)

### 1. Add Unit Tests for MPIStub ⚡ HIGH PRIORITY
**Effort:** 4-6 hours | **Impact:** HIGH

Create `test/test_mpi_stub.py` with tests for:
- `Intracomm` methods (Reduce, Allreduce, Allgather, Barrier)
- `Datatype` constructors (Create_contiguous, Create_vector, Create_subarray)
- `File` operations with various modes
- Error conditions (e.g., root != 0)

**Why:** Currently no direct tests of the stub. This is a maintenance risk.

**Quick Start:**
```bash
# Create test file
touch test/test_mpi_stub.py

# Add basic test structure
cat > test/test_mpi_stub.py << 'EOF'
import pytest
import numpy as np
from NuMPI import MPIStub

def test_reduce_copies_data():
    comm = MPIStub.Intracomm()
    sendbuf = np.array([1.0, 2.0, 3.0])
    recvbuf = np.zeros(3)
    comm.Reduce(sendbuf, recvbuf)
    np.testing.assert_array_equal(recvbuf, sendbuf)

def test_reduce_rejects_nonzero_root():
    comm = MPIStub.Intracomm()
    with pytest.raises(ValueError, match="Root must be zero"):
        comm.Reduce(np.array([1.0]), np.zeros(1), root=1)

# TODO: Add more tests
EOF

# Run tests
pytest test/test_mpi_stub.py -v
```

---

### 2. Improve Documentation ⚡ HIGH PRIORITY
**Effort:** 2-3 hours | **Impact:** HIGH

Update `NuMPI/MPIStub.py` docstring (line 26-33) to include:
- Complete list of supported operations
- Complete list of unsupported operations
- Usage examples
- Semantic differences from real MPI

**Why:** Users can't tell what's supported without reading code.

**Template:** See detailed docstring in `MPI_STUB_ROADMAP.md` Priority 1.2

---

## Quick Wins (Do This Month)

### 3. Fix FIXME for File Modes
**Effort:** 2-3 hours | **Impact:** MEDIUM

Implement missing file opening modes in `NuMPI/MPIStub.py:305-336`:
- `MODE_EXCL`
- `MODE_DELETE_ON_CLOSE`
- `MODE_UNIQUE_OPEN`
- `MODE_SEQUENTIAL`
- `MODE_APPEND`

**Why:** Low-hanging fruit, improves standard conformance.

---

### 4. Add `info` Parameter to File.Open()
**Effort:** 30 minutes | **Impact:** LOW

Add optional `info` parameter to `File.Open()` (line 414). Accept but ignore it (appropriate for stub).

```python
@classmethod
def Open(cls, comm, filename, amode=MODE_RDONLY, info=None):
    # info parameter accepted but ignored in stub
    return File(comm, filename, amode)
```

---

## Future-Proofing (Do Next Quarter)

### 5. Implement Bcast()
**Effort:** 30 minutes | **Impact:** HIGH

Add broadcast operation (no-op for single process, but commonly expected):

```python
def Bcast(self, buf, root=0):
    """Broadcast data from root to all processes."""
    if root != 0:
        raise ValueError("Root must be zero for MPI stub implementation.")
    pass  # No-op in single process

bcast = Bcast
```

**Why:** One of the most common MPI operations. Future code may expect it.

---

### 6. Implement Gather() and Scatter()
**Effort:** 2 hours | **Impact:** MEDIUM

Add non-All versions of gather and scatter operations. See `MPI_STUB_ROADMAP.md` for implementation.

**Why:** Complements existing Allgather, commonly used in MPI codes.

---

### 7. Add Error Messages for Unsupported Methods
**Effort:** 1 hour | **Impact:** MEDIUM

Add stub methods for Send, Recv, Isend, Irecv that raise `NotImplementedError` with helpful messages:

```python
def Send(self, buf, dest, tag=0):
    raise NotImplementedError(
        "Point-to-point communication is not implemented in the MPI stub. "
        "To use Send/Recv, please install mpi4py. Consider using collective "
        "operations like Bcast, Gather, or Allreduce instead."
    )
```

**Why:** Better user experience when trying unsupported features.

---

## Command Cheat Sheet

```bash
# Run only stub tests
pytest test/test_mpi_stub.py -v

# Run all tests without mpi4py (forces stub usage)
pip uninstall mpi4py -y
pytest test/

# Check test coverage
pytest --cov=NuMPI.MPIStub --cov-report=html test/

# Run flake8 on stub
flake8 NuMPI/MPIStub.py

# Check for TODOs/FIXMEs
grep -n "TODO\|FIXME" NuMPI/MPIStub.py
```

---

## Decision Matrix: What to Implement?

| Feature | Effort | Impact | Priority | Reason |
|---------|--------|--------|----------|---------|
| Unit tests | 4-6h | HIGH | ⚡⚡⚡ | No direct tests exist |
| Documentation | 2-3h | HIGH | ⚡⚡⚡ | Users can't tell what's supported |
| Fix file modes | 2-3h | MED | ⚡⚡ | Removes FIXMEs |
| Add info param | 0.5h | LOW | ⚡⚡ | Quick win |
| Bcast() | 0.5h | HIGH | ⚡⚡ | Commonly expected |
| Gather/Scatter | 2h | MED | ⚡ | Future-proofing |
| Error messages | 1h | MED | ⚡ | Better UX |
| Type hints | 3-4h | MED | Future | Nice to have |
| Performance | 6-8h | LOW | Future | Works fine today |

---

## When NOT to Implement

**Don't implement these** (out of scope for a serial stub):
- ❌ Send/Recv point-to-point communication
- ❌ Non-blocking operations (Isend/Irecv/Request)
- ❌ Communicator management (Dup/Split/Create)
- ❌ Groups or Topologies
- ❌ Multi-process support

**Reason:** This is a *serial execution stub*, not a full MPI implementation. These features require actual parallelism.

---

## Getting Started Right Now

**5-Minute Quick Start:**

1. Create test file:
   ```bash
   touch test/test_mpi_stub.py
   ```

2. Add one simple test:
   ```python
   import numpy as np
   from NuMPI import MPIStub

   def test_stub_rank_is_zero():
       comm = MPIStub.COMM_WORLD
       assert comm.Get_rank() == 0
       assert comm.rank == 0
   ```

3. Run it:
   ```bash
   pytest test/test_mpi_stub.py -v
   ```

4. See it pass ✓

5. Add more tests! Target: 20+ tests covering all methods.

---

## Questions?

- **Why prioritize testing?** Because modifying code without tests is risky.
- **Why not add all MPI methods?** Because it's a serial stub, not a parallel library.
- **Should we match mpi4py exactly?** For supported methods, yes. For unsupported, raise clear errors.
- **What if I need a missing method?** Either implement it (for collective ops) or install mpi4py (for point-to-point).

---

## See Also

- `MPI_STUB_ROADMAP.md` - Detailed roadmap with implementation examples
- `NuMPI/MPIStub.py` - Current implementation
- `test/IO/test_io.py` - Examples of stub usage in tests
