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
"""
serial test
"""

import numpy as np
import pytest

try:
    import scipy.optimize

    _scipy_present = True
except ModuleNotFoundError:
    _scipy_present = False

if _scipy_present:
    try:
        from scipy.optimize._linesearch import scalar_search_wolfe2
    except ModuleNotFoundError:
        from scipy.optimize.linesearch import scalar_search_wolfe2

import test.Optimization.MinimizationProblems as mp

from NuMPI import MPI
from NuMPI.Optimization import l_bfgs
from NuMPI.Optimization.Wolfe import (first_wolfe_condition,
                                      second_wolfe_condition)

pytestmark = pytest.mark.skipif(
    MPI.COMM_WORLD.Get_size() > 1,
    reason="tests only serial funcionalities, please execute with pytest")


def my_print(*args):
    # print(*args) # uncomment this line to enable debug prints.
    pass


@pytest.mark.skipif(not _scipy_present, reason='scipy not present')
@pytest.mark.parametrize("Objective", [mp.Trigonometric])
@pytest.mark.parametrize("n", [10, 30])
def test_compare_scipy(Objective, n):
    x0 = Objective.startpoint(n)

    resLBGFS = l_bfgs(
        Objective.f,
        x0,
        args=(2,),
        jac=Objective.grad,
        maxcor=5,
        gtol=1e-8,
        maxiter=1000)
    assert resLBGFS.success
    # Patch
    # resScipy = scipy.optimize.minimize(lambda x: Objective.f(np.reshape(x,
    # (-1,1))),np.reshape(x0,(-1,)), jac = lambda x : np.reshape(
    # Objective.grad(np.reshape(x,(-1,1))),(-1,)),method='L-BFGS-B',tol = 1e-4)
    resScipy = scipy.optimize.minimize(
        Objective.f,
        np.reshape(x0, (-1,)),
        args=(2,),
        jac=Objective.grad,
        method="L-BFGS-B",
        options=dict(gtol=1e-8))
    assert resScipy.success

    np.testing.assert_allclose(np.reshape(resLBGFS.x, (-1,)), resScipy.x,
                               rtol=1e-2,
                               atol=1e-5)  # TODO: The location of the
    # minimum seems o be hard to find
    assert np.abs(Objective.f(resScipy.x, 2) - Objective.f(resLBGFS.x, 2)) < 1e-8


@pytest.mark.skipif(not _scipy_present, reason='scipy not present')
def test_3D():
    # Gaussian Potential

    # quadratic
    toPlot = False

    def ex_fun(x):
        "x should be an np array"
        # x.shape = (-1, 1)
        return np.sum(np.dot((x ** 2).flat, np.array([1, 4, 9])), axis=0)

    def ex_jac(x):
        return 2 * np.array((1 * x[0], 4 * x[1], 9 * x[2]))

    xg, yg = np.linspace(-5, 5, 51), np.linspace(-6, 6, 51)

    def mat_fun(x_g, x_):
        Z = np.zeros((xg.size, yg.size))

        for i in range(Z.shape[0]):
            for j in range(Z.shape[1]):
                Z[j, i] = ex_fun(np.array([xg[i], yg[j], 0]))
        return Z

    # plot
    if toPlot:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        X, Y = np.meshgrid(xg, yg)
        ax.contour(X, Y, mat_fun(xg, yg))
        fig.show()

        my_print(ex_fun(np.array((0, 0, 0))))
        fig2, ax2 = plt.subplots()
        for z in [0]:
            for y in [0, 1, 2]:
                ax2.plot(xg, [ex_fun(np.array((x, y, z))) for x in xg])
                ax2.plot(
                    xg, [ex_jac(np.array((x, y, z)))[0] for x in xg], '--')
        # plt.show(block = True)

    # Initial history:
    x_old = np.array([2., 1., -1.], dtype=float)
    x_old = x_old.reshape(-1, 1)
    # x_old.shape=(-1,1)
    grad_old = ex_jac(x_old)

    # line search
    reslinesearch = scipy.optimize.minimize_scalar(
        fun=lambda alpha: ex_fun(x_old - grad_old * alpha), bounds=(0, 2000),
        method="bounded")
    assert reslinesearch.success
    my_print("alpha {}".format(reslinesearch.x))
    x = x_old - grad_old * reslinesearch.x
    assert ex_fun(x) < ex_fun(x_old)
    # grad = ex_jac(x)
    my_print("x first_linesearch {}".format(x))
    # k = 1
    my_print("Wolfe")

    my_print("1st: {}".format(
        first_wolfe_condition(ex_fun, x_old, ex_jac, -grad_old,
                              reslinesearch.x, beta1=1e-4)))
    my_print("2nd {}".format(
        second_wolfe_condition(x_old, ex_jac, -grad_old, reslinesearch.x,
                               beta2=0.9)))

    resscipy = scipy.optimize.minimize(ex_fun, np.reshape(x, (-1,)), jac=ex_jac,
                                       options=dict(gtol=1e-10))
    my_print("scipy success: {}".format(resscipy.success))
    my_print("scipy nit {}".format(resscipy.nit))
    my_print("scipy result: {}".format(resscipy.x))

    res = l_bfgs(ex_fun, x, jac=ex_jac, x_old=x_old, maxcor=3, maxiter=100,
                 gtol=1e-10, ftol=0)
    assert res.success
    my_print("nit {}".format(res.nit))

    if toPlot:
        import matplotlib.pyplot as plt
        fig3d = plt.figure()
        ax3d = fig3d.add_subplot(111, projection='3d')

        for it, i in zip(res['iterates'], range(len(res['iterates']))):
            ax3d.plot(it.x[0], it.x[1], it.x[2], '+k')
            # ax3d.annotate(i, it.x)
        fig3d.show()
        plt.show(block=True)

    np.testing.assert_almost_equal(res.x, np.zeros(res.x.shape), decimal=5)


@pytest.mark.skipif(not _scipy_present, reason='scipy not present')
def test_quadratic_nD():
    # quadratic
    n = 50

    factors = np.random.random(n) + 0.1

    def ex_fun(x):
        "x should be an np array"
        # x.shape = (-1, 1)
        return np.sum(np.dot((x ** 2).flat, factors ** 2), axis=0)

    def ex_jac(x):
        return 2 * np.diag(factors).dot(x)

    # plot

    # Initial history:
    x_old = np.random.random(n)
    x_old = x_old.reshape(-1, 1)
    # x_old.shape=(-1,1)
    grad_old = ex_jac(x_old)

    # line search
    alpha, phi, phi0, derphi = scalar_search_wolfe2(
        lambda alpha: ex_fun(x_old - grad_old * alpha),
        lambda alpha: np.dot(ex_jac(x_old - grad_old * alpha).T, -grad_old))
    assert derphi is not None
    x = x_old - grad_old * alpha
    assert ex_fun(x) < ex_fun(x_old)
    # grad = ex_jac(x)
    my_print("x first_linesearch {}".format(x))
    # k = 1

    resscipy = scipy.optimize.minimize(ex_fun, np.reshape(x, (-1,)), jac=ex_jac)
    my_print("schipy success: {}".format(resscipy.success))

    # my_print(x)
    # my_print(ex_jac(x))

    res = l_bfgs(ex_fun, x, jac=ex_jac, x_old=x_old, maxcor=5, maxiter=100,
                 g2tol=1e-10)
    my_print("nit {}".format(res.nit))

    if False:
        import matplotlib.pyplot as plt
        fig3d = plt.figure()
        ax3d = fig3d.add_subplot(111, projection='3d')

        for it, i in zip(res['iterates'], range(len(res['iterates']))):
            ax3d.plot(it.x[0], it.x[1], it.x[2], '+k')
            # ax3d.annotate(i, it.x)
        fig3d.show()
        plt.show(block=True)

    np.testing.assert_almost_equal(res.x, np.zeros((n, 1)), decimal=4)


@pytest.mark.parametrize("n", [3, 5, 10,
                               20])  # This has Problems with Linesearch at
# high number of points
def test_gaussian_nD(n):
    factors = np.random.random(n) + 0.5
    shift = (np.random.random(n) - 0.5) * 0.1
    shift = shift.reshape(-1, 1)

    def ex_fun(x):
        "x should be an np array"
        # x.shape = (-1, 1)
        return - np.exp(
            - np.sum(np.dot(((x - shift) ** 2).flat, factors ** 2), axis=0))

    def ex_jac(x):
        return - 2 * np.diag(factors).dot(x - shift) * ex_fun(x)

    # plot

    # Initial history:
    x = np.random.random(n) - 0.5
    x = x.reshape(-1, 1)

    # resscipy = scipy.optimize.minimize(ex_fun, x, jac=ex_jac)
    # my_print("schipy success: {}".format(resscipy.success))

    res = l_bfgs(ex_fun, x, jac=ex_jac, maxcor=2, gtol=1e-5, maxiter=10000)
    my_print("nit {}".format(res.nit))

    if False:
        import matplotlib.pyplot as plt
        fig3d = plt.figure()
        ax3d = fig3d.add_subplot(111, projection='3d')

        for it, i in zip(res['iterates'], range(len(res['iterates']))):
            ax3d.plot(it.x[0], it.x[1], it.x[2], '+k')
            # ax3d.annotate(i, it.x)
        fig3d.show()
        plt.show(block=True)

    np.testing.assert_allclose(res.x, shift, atol=1e-3)


@pytest.mark.skipif(not _scipy_present, reason='scipy not present')
def test_x2_xcosy():
    def ex_fun(x_):
        x = x_.reshape((-1, 1))
        return .5 * x[0, 0] ** 2 + x[0, 0] * np.cos(x[1, 0])

    def ex_jac(x_):
        x = x_.reshape((-1, 1))
        return np.array([[x[0, 0] + np.cos(x[1, 0])],
                         [-x[0, 0] * np.sin(x[1, 0])]])

    # plot

    # plt.show(block=True)
    ######
    # Initial history:
    x = np.array([[3], [-4]], dtype=float)

    # k = 1

    res = l_bfgs(ex_fun, x, jac=ex_jac, gtol=1e-5, maxcor=5, maxiter=10000,
                 linesearch_options=dict(c1=1e-4, c2=0.999))
    my_print("nit {}".format(res.nit))

    ref = scipy.optimize.minimize(ex_fun, x.reshape(-1),
                                  jac=lambda x: ex_jac(x).reshape(-1))

    assert ref.success

    if False:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        xg, yg = np.linspace(-5, 5, 51), np.linspace(-6, 6, 51)

        def mat_fun(xg, yg):
            Z = np.zeros((xg.size, yg.size))
            for i in range(Z.shape[0]):
                for j in range(Z.shape[1]):
                    Z[j, i] = ex_fun(np.array([xg[i], yg[j]]))
            return Z

        X, Y = np.meshgrid(xg, yg)
        plt.colorbar(ax.contour(X, Y, mat_fun(xg, yg)))
        fig.show()
        for it, i in zip(res['iterates'], range(len(res['iterates']))):
            ax.plot(it.x[0], it.x[1], '+k')
            ax.annotate(i, it.x)
        plt.show(block=True)

    np.testing.assert_allclose(res.x.reshape(-1), ref.x, atol=1e-16, rtol=1e-5)
