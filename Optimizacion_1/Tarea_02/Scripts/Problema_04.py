import numpy as np


def dx(x, y):
    return (12 * x**3 - 12 * x**2 - 24 * x) / (12 * (1 + 4 * y**2))


def dy(x, y):
    return (3 * x**4 - 4 * x**3 - 12 * x**2 + 18) / (-12 *
                                                     (1 + 4 * y**2)) * (8 * y)


def dyy(x, y):
    return 8 * (3 * x**4 - 4 * x**3 - 12 * x**2 +
                18) / (12) * (4 * y**2 - 1) / ((4 * y**2 + 1)**2)


def dxx(x, y):
    return (36 * x**2 - 24 * x - 24) / (12 * (1 + 4 * y**2))


def dxy(x, y):
    return (12 * x**3 - 12 * x**2 - 24) / (-12 * (1 + 4 * y**2)**2) * (8 * y)


values = [[0, 0], [2, 0], [-1, 0], [1.20113, 0], [2.54167, 0]]
for value in values:
    x, y = value
    print("------------------------------------------------------------")
    print("(x,y):\t({},{})".format(x, y))
    print("dxx:{: >35}".format(dxx(x, y)))
    print("dyy:{: >35}".format(dyy(x, y)))
    print("dxy:{: >35}".format(dxy(x, y)))
    print("dxx*dxy-dxy**2:\t{: >23}".format(
        dxx(x, y) * dyy(x, y) - (dxy(x, y)**2)))
    print("------------------------------------------------------------\n")
