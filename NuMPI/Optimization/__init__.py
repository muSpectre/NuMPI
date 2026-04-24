#
# Copyright 2018, 2020 Antoine Sanner
#           2019, 2026 Lars Pastewka
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
MPI-parallel optimization routines.

Public entry points
-------------------
- ``l_bfgs``            : unconstrained L-BFGS with a Wolfe line search.
- ``l_bfgs_projected``  : projected L-BFGS on a single linear equality plus
                          optional box bounds (uses Armijo backtracking with
                          projection arc). Handles the same feasible set as
                          ``constrained_conjugate_gradients``.
- ``constrained_conjugate_gradients`` : bound-constrained CG (Bugnicourt
                          et al. 2018), optionally with a linear equality
                          via the ``linear_constraint`` kwarg or the legacy
                          ``mean_val`` shortcut.
- ``LinearConstraint``  : utility encapsulating a single affine constraint
                          ``<a, x> = target`` with multiplier / tangent /
                          Euclidean-projection operations.

MPI contract (all optimizers)
-----------------------------
Arrays are **distributed**: each rank holds a slice of the global ``x``,
gradient, bounds, ``LinearConstraint.a``, and any other ``N``-length data.
Scalars are **global**: energies, tolerances, and multipliers have the same
value on every rank.

The single load-bearing requirement on the caller is that the objective
function return a **globally reduced** scalar energy:

    def fun(x):
        local = 0.5 * np.sum((x - y_local) ** 2)   # local contribution
        return pnp.sum(local).item(), (x - y_local)   # scalar REDUCED, grad local

where ``pnp = Reduction(comm)``. Returning a per-rank local energy leads to
silently divergent trajectories across ranks (line-search acceptance tests
disagree and different ranks end up at different iterates).

``callback(x)`` receives the *local* slice. Gather yourself if you need the
global state for I/O.

See the "MPI Conventions" section of the top-level ``README.md`` for a fuller
treatment and a worked example.
"""


from .LBFGS import l_bfgs  # noqa F401
from .LinearConstraint import LinearConstraint  # noqa F401
from .ProjectedLBFGS import l_bfgs_projected  # noqa F401