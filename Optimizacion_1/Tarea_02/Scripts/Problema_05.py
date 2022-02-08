import matplotlib.pyplot as plt
import numpy as np


def function(x1, x2):
    return 8 * x1 + 12 * x2 + x1**2 - 2 * x2**2


parameters = {
    0: {
        "color": "#390099",
        "level": -10,
    },
    1: {
        "color": "#9e0059",
        "level": -5,
    },
    2: {
        "color": "#ff0054",
        "level": 0,
    },
    3: {
        "color": "#ff5400",
        "level": 5,
    },
    4: {
        "color": "#ffbd00",
        "level": 10,
    }
}

x1 = np.linspace(-12, 4, 1000)
x2 = np.linspace(-2, 8, 1000).reshape(-1, 1)
z = function(x1, x2)
x1, x2 = np.meshgrid(x1, x2)
plots = []
labels = []
plt.subplots(figsize=(8, 6))
for parameter in parameters:
    dataset = parameters[parameter]
    label = "level={}".format(dataset["level"])
    cs = plt.contour(x1,
                     x2,
                     z,
                     linestyles="-",
                     levels=[dataset["level"]],
                     colors=dataset["color"])
    legend, _ = cs.legend_elements()
    plots += legend
    labels += [label]
plt.legend(plots,
           labels,
           ncol=len(plots),
           frameon=False,
           bbox_to_anchor=(0.92, 1.075))
plt.grid(
    ls="--",
    color="#000000",
    alpha=0.5,
)
plt.tight_layout()
plt.savefig("../Document/Graphics/problem05.png")
