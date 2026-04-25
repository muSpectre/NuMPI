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


import warnings

try:
    # raise ImportError()
    from mpi4py import MPI

    _has_mpi4py = True
except ImportError:
    warnings.warn('Could not import mpi4py; '
                  'falling back to MPI stub implementation.', ImportWarning)
    from . import MPIStub as MPI  # noqa: F401

    _has_mpi4py = False

from . import Optimization, Tools  # noqa: F401

try:
    from ._version import __version__
except ImportError:
    # _version.py is written by hatch-vcs at build time. If it's absent
    # (e.g. running from an unbuilt source tree), fall back to installed
    # distribution metadata.
    try:
        from importlib.metadata import PackageNotFoundError, version

        __version__ = version("NuMPI")
    except (ImportError, PackageNotFoundError):
        __version__ = "0.0.0+unknown"
