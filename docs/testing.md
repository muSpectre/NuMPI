# Testing utilities

`NuMPI.Testing` helps to write pytest suites that run serially and under
`mpirun`, with or without mpi4py.

## Communicator fixtures

```python
from NuMPI.Testing.Fixtures import MPITestFixture
```

`MPITestFixture(commsize, scope="function")` creates a pytest fixture that is
parametrized over the communicator sizes in `commsize`. For each size `n`, the
first `n` processes run the test with a communicator of size `n`; the other
processes, and all processes if there are fewer than `n`, skip it. Define the
fixtures in `conftest.py`:

```python
# conftest.py
from NuMPI.Testing.Fixtures import MPITestFixture

comm = MPITestFixture([1, 2, 4], scope="session")
comm_self = MPITestFixture([1], scope="session")
```

```python
# test_example.py
import numpy as np
from NuMPI.Tools import Reduction

def test_sum(comm):
    pnp = Reduction(comm)
    assert pnp.sum(np.ones(10)) == 10 * comm.size
```

`pytest` runs `test_sum` with a communicator of size 1 and skips sizes 2 and
4; `mpirun -n 4 python -m pytest` runs it with all three sizes. Without mpi4py,
the fixture yields the [MPI stub](mpi.md#the-mpi-stub)'s communicator and runs
only size 1.

## Serial-only tests

```python
from NuMPI.Testing.Fixtures import serial_only

@serial_only
def test_something_serial():
    ...

pytestmark = serial_only  # for all tests of a module
```

`serial_only` skips a test when it runs on more than one process, e.g. tests
that write files with fixed names or only test serial functionality.

## Parallel assertions

```python
from NuMPI.Testing.Assertions import (
    parallel_assert, assert_all_allclose, assert_all_array_equal,
    assert_one_allclose, assert_one_array_equal,
)
```

An assertion that fails on only some processes leaves the others waiting in the
next collective operation. These assertions fail on all processes if they fail
on any:

| Function | Checks |
|----------|--------|
| `parallel_assert(comm, condition, message)` | `condition` on every process |
| `assert_all_allclose(comm, actual, desired, ...)` | `np.testing.assert_allclose` on every process |
| `assert_all_array_equal(comm, actual, desired, ...)` | `np.testing.assert_array_equal` on every process |
| `assert_one_allclose(comm, rank, actual, desired, ...)` | `np.testing.assert_allclose` on process `rank` only |
| `assert_one_array_equal(comm, rank, actual, desired, ...)` | `np.testing.assert_array_equal` on process `rank` only |

```python
import numpy as np
from NuMPI.Testing.Assertions import assert_one_array_equal

def test_save(comm):
    ...  # save_npy("out.npy", ...) on all processes
    comm.barrier()
    data = np.load("out.npy") if comm.rank == 0 else None
    assert_one_array_equal(comm, 0, data, expected)
```
