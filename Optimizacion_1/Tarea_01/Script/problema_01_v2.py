import matplotlib.pyplot as plt
import numpy as np


def f1(x, y):
    return x**2 - y**2


def f2(x, y):
    return 2 * x * y


x = np.linspace(-10, 10, 100)
y = x.reshape(-1, 1)
z1 = f1(x, y)
z2 = f2(x, y)
x, y = np.meshgrid(x, y)
plt.xlim(-10, 10)
plt.ylim(-10, 10)
plt.xticks(np.arange(-10, 12, 2))
plt.yticks(np.arange(-10, 12, 2))
cs1 = plt.contour(x, y, z1, levels=[12],
                  colors="#b5179e")
cs2 = plt.contour(x, y, z2, levels=[16],
                  colors="#3a0ca3")
plt.grid(ls="--", color="#000000", alpha=0.5)
plt.tight_layout()
plt.show()
