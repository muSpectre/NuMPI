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


import numpy as np


def first_wolfe_condition(fun, x0, fprime, direction, alpha, beta1):
    """
    p. 268, 11.19

    Keyword Arguments:
    fun         -- objective function to minimize
    x0          -- initial guess for solution
    fprime      -- Jacobian (gradient)
    direction   -- search direction (column vec)
    alpha       -- step size
    beta1       -- lower wolfe bound
    """
    return (fun(x0 + alpha * direction) <= fun(x0) +
            alpha * beta1 * np.dot(fprime(x0).T, direction).item())


def second_wolfe_condition(x0, fprime, direction, alpha, beta2):
    """
    p. 270, 11.21

    Keyword Arguments:
    x0        -- initial guess for solution
    fprime    -- Jacobian (gradient) of objective function
    direction -- search direction
    alpha     -- step size
    beta2     -- upper wolfe bound
    """
    return (np.dot(fprime(x0 + alpha * direction).T, direction).item() >=
            beta2 * np.dot(fprime(x0).T, direction).item())
