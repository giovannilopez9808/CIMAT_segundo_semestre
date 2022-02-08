import numpy as np


def dx1(x1, x2):
    return -200*(x2-x1**2)*2*x1-2*(1-x1)*x1


def dx2(x1, x2):
    return 200*(x2-x1**2)


def dx1x1(x1, x2):
    return 1200*x1**2+4*x1-400*x2-2


def dx2x2(x1, x2):
    return 200


def dx1x2(x1, x2):
    return -400*x1


values = [[0, 0], [1, 1]]
for value in values:
    x, y = value
    print("------------------------------------------------------------")
    print("(x,y):\t({},{})".format(x, y))
    print("dxx:\t{}".format(dx1x1(x, y)))
    print("dyy:\t{}".format(dx2x2(x, y)))
    print("dxy:\t{}".format(dx1x2(x, y)))
    print("dxx*dxy-dxy**2:\t{}".format(
        dx1x1(x, y) * dx2x2(x, y) - (dx1x2(x, y)**2)))
    print("------------------------------------------------------------\n")
