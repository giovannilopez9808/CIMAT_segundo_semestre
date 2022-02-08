import matplotlib.pyplot as plt
import numpy as np


def function(x: np.array, y: np.array) -> np.array:
    f = 100 * (y - x**2)**2 + (1 - x**2)**2
    return f


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


parameters = {
    0: {
        "color": "#ffbd00",
        "level": 1,
    },
    1: {
        "color": "#ff5400",
        "level": 3,
    },
    2: {
        "color": "#ff0054",
        "level": 5,
    },
    3: {
        "color": "#9e0059",
        "level": 7,
    },
    4: {
        "color": "#390099",
        "level": 9,
    }
}

x = np.linspace(-5, 5, 1000)
y = np.delete(x, 0)
x = np.delete(x, -1)
y = y.reshape(-1, 1)
z = function(x, y)
x, y = np.meshgrid(x, y)
plots = []
labels = []
plt.subplots(figsize=(9, 6))
for parameter in parameters:
    dataset = parameters[parameter]
    level = dataset["level"]
    color = dataset["color"]
    label = "level={}".format(level)
    cs = plt.contour(x, y, z, linestyles="-", levels=[level], colors=color)
    legend, _ = cs.legend_elements()
    plots += legend
    labels += [label]
plt.xlim(-2, 2)
plt.ylim(-1, 5)
plt.legend(plots,
           labels,
           ncol=len(plots),
           frameon=False,
           fontsize=12,
           bbox_to_anchor=(0.95, 1.07))
# plt.grid(ls="--",
#          color="#000000",
#          alpha=0.5)
plt.tight_layout()
plt.savefig("../Document/Graphics/problem06.png")


values = [[0, 0], [1, 1]]
for value in values:
    x, y = value
    print("------------------------------------------------------------")
    print("(x,y):\t({},{})".format(x, y))
    print("dxx:{: >20}".format(dx1x1(x, y)))
    print("dyy:{: >20}".format(dx2x2(x, y)))
    print("dxy:{: >20}".format(dx1x2(x, y)))
    print("dxx*dxy-dxy**2:{: >9}".format(
        dx1x1(x, y) * dx2x2(x, y) - (dx1x2(x, y)**2)))
    print("------------------------------------------------------------\n")
