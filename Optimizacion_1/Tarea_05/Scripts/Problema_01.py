from tabulate import tabulate
from numpy import array
from numpy.linalg import norm


def fx(x: array) -> float:
    x1, x2 = x
    result = (x1**2 + x2**2 - 1)**2 + (x2**2 - 1)**2
    return result


def dfx1(x: array) -> float:
    x1, x2 = x
    result = 4 * x1 * (x1**2 + x2**2 - 1)
    return result


def dfx2(x: array) -> float:
    x1, x2 = x
    result = 4 * x2 * (x1**2 + x2**2 - 1) + 4 * (x2**2 - 1) * x2
    return result


def ddfx1(x: array) -> float:
    x1, x2 = x
    result = 4 * (x1**2 + x2**2 - 1) + 8 * x1**2
    return result


def ddfx2(x: array) -> float:
    x1, x2 = x
    result = 8 * x2 + 4 * (x1**2 + x2**2 - 1) + 4 * (x2**2 - 1) + 8 * x2**2
    return result


def ddfx1x2(x: array) -> float:
    x1, x2 = x
    result = 8 * x1 * x2
    return result


points = array([[0, 1], [0, -1], [1, 0], [-1, 0]])
results = []
for x in points:
    fx1 = fx(x)
    grad = array([dfx1(x), dfx2(x)])
    hess = ddfx1(x) * ddfx2(x) - (ddfx1x2(x))**2
    results += [[x[0], x[1], fx1, norm(grad), hess]]
print(
    tabulate(results,
             headers=[
                 "x1",
                 "x2",
                 "Funcion",
                 "Gradiente",
                 "Hessiano",
             ]))
