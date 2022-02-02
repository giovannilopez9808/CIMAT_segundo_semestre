import matplotlib.pyplot as plt
import numpy as np


def f1(x, y):
    return x**2 - y**2


def f2(x, y):
    return 2 * x * y


parameters = {"lower": -500, "upper": 500}
x = np.linspace(parameters["lower"], parameters["upper"], 1000)
y = np.linspace(parameters["lower"], parameters["upper"], 1000)
x_sol = []
y_sol = []
for xi in x:
    for yi in y:
        f1xy = f1(xi, yi)
        f2xy = f2(xi, yi)
        if abs(f1xy - 12) < 1e-5:
            x_sol.append(xi)
            y_sol.append(yi)
plt.plot(x_sol, y_sol)
plt.show()
