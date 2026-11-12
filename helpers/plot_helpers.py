#
# Copyright 2018 Antoine Sanner
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
from matplotlib.patches import FancyArrowPatch

rec = 0


def asscalar(a):
    #    global rec
    #    print(rec)
    #    rec=rec+1;
    #    print(a)
    if isinstance(a, np.ndarray):
        return asscalar(a.tolist()[0])
    elif isinstance(a, (list, tuple)):
        return asscalar(a[0])
    else:
        return a


def draw_npArrow2D(ax, start, end=None, delta=None, arrowprops={}, label=None, textprops={}):
    arrow_prop_dict = dict(mutation_scale=10, arrowstyle='-|>', color='k', shrinkA=0, shrinkB=0)
    for key in arrowprops:
        arrow_prop_dict[key] = arrowprops[key]

    xs = asscalar(start[0])
    ys = asscalar(start[1])

    if end is not None:
        xe = asscalar(end[0])
        ye = asscalar(end[1])
    elif delta is not None:
        xe = xs + asscalar(delta[0])
        ye = ys + asscalar(delta[1])
    else:
        raise AssertionError('Missing Argument end or delta')

    ax.add_artist(FancyArrowPatch((xs, ys), (xe, ye), **arrow_prop_dict))
    if label is not None:
        ax.text(xe, ye, label, **textprops)
