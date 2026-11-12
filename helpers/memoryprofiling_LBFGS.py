#
# Copyright 2020 Antoine Sanner
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

import sys
from test.Optimization.MPIMinimizationProblems import MPI_Quadratic

from memory_profiler import profile

from NuMPI import MPI
from NuMPI.Optimization import l_bfgs
from NuMPI.Tools import Reduction

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nprocs = comm.Get_size()
pnp = Reduction(comm)
print(sys.argv)
n = int(sys.argv[1])
print(n)

fp = open("LBFGS_memory_profile_n{}_nprocs{}_rank{}.log".format(n, nprocs, rank), "w")
LBFGS = profile(stream=fp)(l_bfgs)

Objective = MPI_Quadratic(n, pnp=pnp)
result = LBFGS(Objective.f, Objective.startpoint(),
               jac=Objective.grad,
               options={"gtol": 1e-6, "pnp": Objective.pnp})
fp.close()
assert result.success
print(result.nit)
