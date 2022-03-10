from numpy import linspace, meshgrid, arange
import matplotlib.pyplot as plt


def f(x, y):
    fx = (x**2 + y**2 - 1)**2 + (y**2 - 1)**2
    return fx


parameters = {
    "path graphics": "../Graphics/",
    "file graphics": "Problema_01.png"
}
x = linspace(-1, 1, 100)
y = x.reshape(-1, 1)
z = f(x, y)
x, y = meshgrid(x, y)
#plt.xticks(arange(-1, 1.25, 0.25))
#plt.yticks(arange(-1, 1.25, 0.25))
plt.scatter(
    [0, 0, 0, 1, -1],
    [1, -1, 0, 0, 0],
    marker="*",
    s=100,
    color="green",
)
plt.contour(x, y, z, cmap="inferno")
plt.grid(ls="--", color="#000000", alpha=0.5)
plt.tight_layout()
plt.savefig("{}{}".format(parameters["path graphics"],
                          parameters["file graphics"]))
