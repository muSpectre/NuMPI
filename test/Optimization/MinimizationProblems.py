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

"""
These are Testfunctions extracted from

Mori, J. J., Garbow, B. S. & Hillstrom, K. E. Testing Unconstrained
 Optimization Software. 25 (1981).

In future I may use Scipy's example functions, see
 https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.rosen
.html#scipy.optimize.rosen
"""

import abc


class ObjectiveFunction(object, metaclass=abc.ABCMeta):

    @staticmethod
    @abc.abstractmethod
    def f_grad(x):
        pass

    @classmethod
    def f(cls, x, *args):
        return cls.f_grad(x, *args)[0]

    @classmethod
    def grad(cls, x, *args):
        return cls.f_grad(x, *args)[1]

    @classmethod
    def plot_2D(cls):
        import matplotlib.pyplot as plt

        from helpers.plot_helpers import draw_npArrow2D
        fun = cls.f

        xg, yg = np.linspace(*cls.bounds, 51), np.linspace(*cls.bounds, 51)

        def mat_fun(xg, yg, fun):
            Z = np.zeros((xg.size, yg.size))

            for i in range(Z.shape[0]):
                for j in range(Z.shape[1]):
                    Z[j, i] = fun(np.array([xg[i], yg[j]]))
            return Z

        X, Y = np.meshgrid(xg, yg)

        plt.colorbar(
            plt.pcolormesh(X, Y, np.log10(mat_fun(xg, yg, fun)))).set_label(
            r"log10($\phi$)")
        plt.plot(1 / 2, 1 / 2, '+')  # standart starting point
        ax = plt.gca()
        for x in [np.array([[0], [1]]),
                  np.array([[-2], [-2]]),
                  np.array([[2], [-3]]),
                  np.array([[-3], [1]])]:
            draw_npArrow2D(ax, x, delta=cls.grad(x) / 10)

            eps = 1e-4
            # unit Vector with random Direction
            phi = (np.random.random(1) * 2 * np.pi).item()
            u = np.array([[np.cos(phi)], [np.sin(phi)]])

            print((cls.f(x + u * eps) - cls.f(x)) / eps)
            print(cls.grad(x).T @ u)

        plt.show(block=True)


class Trigonometric(ObjectiveFunction):
    """
    all values of n are allowed

    :param x: 1d array
    :return:
    """
    bounds = (-np.pi, np.pi)

    @staticmethod
    def f_grad(x_, arg1):
        assert arg1 == 2
        n = x_.size
        old_shape = x_.shape
        x = np.reshape(x_, (-1, 1))
        idxVector = np.reshape(np.arange(1, n + 1), (-1, 1))
        f = n - np.sum(np.cos(x)) + idxVector * (1 - np.cos(x)) - np.sin(x)

        jac = np.reshape(np.sin(x), (1, -1)) + np.diag(
            (-np.cos(x) + idxVector * np.sin(x)).flat)
        return np.sum(f ** 2).item(), np.reshape(2 * jac.T @ f, old_shape)

    @staticmethod
    def startpoint(n):
        """
        standard starting point
        :param x:
        :return:
        """

        return np.reshape(np.ones(n) / n, (-1, 1))

    @staticmethod
    def plot_2D():
        import matplotlib.pyplot as plt

        from helpers.plot_helpers import draw_npArrow2D
        fun = Trigonometric.f

        xg, yg = np.linspace(-np.pi, np.pi, 51), np.linspace(-np.pi, np.pi, 51)

        def mat_fun(xg, yg, fun):
            Z = np.zeros((xg.size, yg.size))

            for i in range(Z.shape[0]):
                for j in range(Z.shape[1]):
                    Z[j, i] = fun(np.array([xg[i], yg[j]]))
            return Z

        X, Y = np.meshgrid(xg, yg)

        plt.colorbar(plt.contour(X, Y, mat_fun(xg, yg, fun)))
        plt.plot(1 / 2, 1 / 2, '+')  # standart starting point
        ax = plt.gca()
        for x in [np.array([[0], [1]]),
                  np.array([[-2], [-2]]),
                  np.array([[2], [-3]]),
                  np.array([[-3], [1]])]:
            draw_npArrow2D(ax, x, delta=Trigonometric.grad(x) / 10)

            eps = 1e-4
            # unit Vector with random Direction
            phi = (np.random.random(1) * 2 * np.pi).item()
            u = np.array([[np.cos(phi)], [np.sin(phi)]])

            print((Trigonometric.f(x + u * eps) - Trigonometric.f(x)) / eps)
            print(Trigonometric.grad(x).T @ u)

            Trigonometric.grad(x)

        plt.show(block=True)


class Extended_Rosenbrock(ObjectiveFunction):
    """

    https://github.com/cjtonde/optimize_rosenbrock/blob/master/src
    /optimize_rosenbrock.py

    n should be even

    :param x: 1d array
    :return:
    """
    bounds = (-4, 4)

    @staticmethod
    def f_grad(x, arg1):
        assert arg1 == 2

        sumf2 = (np.sum(
            100 * (x[1::2] - x[:-1:2] ** 2) ** 2 + (1 - x[:-1:2]) ** 2)).item()

        grad = np.zeros_like(x)
        grad[1::2] = 200 * (x[1::2] - x[:-1:2] ** 2)
        grad[:-1:2] = -400 * x[:-1:2] * (x[1::2] - x[:-1:2] ** 2) - 2 * (
                1 - x[:-1:2])  # # d / dx2l-1

        return sumf2, grad

    @staticmethod
    def startpoint(n):
        """
        standard starting point
        :param n:
        :return: array of shape (1,n)
        """
        x0 = np.zeros(n, dtype=float)
        x0 = x0.reshape(-1, 1)
        x0[:-1:2] = -1.2
        x0[1::2] = 1

        return x0

    @staticmethod
    def minVal(*args):
        return 0

    @staticmethod
    def xmin(n):
        """
        Location of minimum according to

        Mori, J. J., Garbow, B. S. & Hillstrom, K. E. Testing Unconstrained
        Optimization Software. 25 (1981).

        This function not necessarily have only one Minimum in higher
        dimensional Space: see e.g. 10.1162/evco.2006.14.1.119

        :param n: number of DOF
        :return: array of size n
        """

        return np.ones((n, 1), dtype=float)


if __name__ == "__main__":
    Extended_Rosenbrock.plot_2D()
    Trigonometric.plot_2D()
