Change log for NuMPI
===================

v0.7.3 (10Apr25)
----------------

- MAINT: Moved `suggest_subdivision` to main package

v0.7.2 (26Mar25)
----------------

- API: Allow passing communicator directly to `l_bfgs`
- BUG: Fixed parallel write into NPY files when a subdomain is only a single
  grid point in one dimension
- MAINT: Return last result when the line search fails
- MAINT: Throw proper errors and report convergence

v0.7.1 (18Mar25)
----------------

- BUG: Fixed reduce operation of arrays with two entries

v0.7.0 (12Mar25)
----------------

- ENH: Writing and reading decomposed arrays of arbitrary dimension to NPY

v0.6.1 (23Jul24)
----------------

- MAINT: Compatibility with numpy 2.0

v0.6.0 (23Jul24)
----------------

- MAINT: Removed `NCStructurdGrid` and dependency on NetCDF4 which caused many
  problems downstream

v0.5.0 (14Apr24)
----------------

- ENH: Added support for passing optional arguments to objective functions
  of optimizers
- BUG: Keep shape of optimization target when passing to objective function and
  jacobian
- MAINT: Switched to `DiscoverVersion` package for version discovery
- MAINT: Refactored file names

v0.4.0 (14Jan24)
----------------

- MAINT: Switched build system to `flit`

v0.3.1 (07Nov22)
----------------

- MAINT: Made `scipy` dependency optional

v0.3.0 (11Nov21)
----------------

- ENH: Support for masked arrays

v0.2.0 (28Oct21)
----------------

- ENH: Reduction for computing arithmetic mean

v0.1.7 (22Jun21)
----------------

- Drop support for python3.6 and take python3.9 in testing
- Fix bug in CI installation

v0.1.6 (22Jun21)
----------------

- ENH: Optimization: parallelized the bound constrained conjugate gradient without restart
- ENH: Optimization: implement bound constrained conjugate gradient without restart when the constrained set changes
- ENH: Optimization: implement bound constrained conjugate gradient with restart when the constrained set changes

v0.1.5 (02Mar21)
----------------

- ENH: NetCDF IO : NCStructuredGrid migrated from muspectre to here
- Drop support for Python3.5

v0.1.4 (16Oct20)
----------------
- BUG: don't close filestream on failure

v0.1.3 (14Oct20)
----------------
- BUG: make_mpi_file_view was not compatible with filestreams

v0.1.2 (29Jun20)
----------------
- Remove usage of xrange (#29)

v0.1.1 (02Dec19)
----------------

- Reader from Stub MPI communicator compatible with filestream input 

v0.1.0 (25Aug19)
----------------

- Initial release
- Stub MPI communicator
- Helper functions for make reduction operations more numpy-like
- MPI-parallel IO of `npy` files
- MPI-parallel L-BFGS
