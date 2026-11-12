#
# Copyright 2019-2020 Antoine Sanner
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
This runs LBFGS and scipy L-BFGS-B in serial and compare the number of iterations needed
"""

import os
from test.Optimization.MinimizationProblems import Extended_Rosenbrock, Trigonometric

import matplotlib.pyplot as plt
import numpy as np
import scipy

from NuMPI.Optimization import l_bfgs

# import sys
n = 10 ** 3


class decorated_objective:
    def __init__(self, objective):
        self.objective = objective
        self.neval = 0
        self.energies = []
        self.maxgradients = []

    def __call__(self, *args, **kwargs):
        f, grad = self.objective(*args, **kwargs)
        self.neval += 1
        self.energies.append(f)
        self.maxgradients.append(np.max(abs(grad)))
        return f, grad


for Objective in [Trigonometric, Extended_Rosenbrock]:
    fig, (axEn, axgrad) = plt.subplots(2, 1, sharex=True)
    for method, name, c in zip([l_bfgs, "L-BFGS-B"], ["NuMPI", "scipy"], ["C1", "C0"]):
        objective_monitor = decorated_objective(Objective.f_grad)
        result = scipy.optimize.minimize(objective_monitor, Objective.startpoint(n),
                                         jac=True, method=method,
                                         options={"gtol": 1e-6, "ftol": 1e-20, "maxcor": 20})
        assert result.success

        axgrad.plot(range(objective_monitor.neval),
                    objective_monitor.maxgradients, label="{}".format(name))
        axEn.plot(
            range(objective_monitor.neval),
            (np.array(objective_monitor.energies) - objective_monitor.energies[-1]) / (
                    objective_monitor.energies[0] - objective_monitor.energies[-1]),
            label="{}".format(name), c=c)

        print("{}, {}: nevals {}, nit {}, evals/it {}"
              .format(name,
                      Objective.__name__,
                      objective_monitor.neval,
                      result.nit, float(objective_monitor.neval) / result.nit))

    axEn.set_xlabel("# of objective evaluations")
    axEn.set_ylabel("E(i)-E(last) / (E(0)-E(last))")
    axEn.set_yscale("log")

    axgrad.set_yscale("log")
    axgrad.set_ylabel(r"$|grad|_{\infty}$")
    axgrad.legend()

    for a in (axEn, axgrad):
        a.set_xlabel("# of objective evaluations")
        a.label_outer()

    fig.suptitle("{}, n={}".format(Objective.__name__, n))
    fig.savefig("{}_{}.png".format(os.path.basename(__file__), Objective.__name__))
