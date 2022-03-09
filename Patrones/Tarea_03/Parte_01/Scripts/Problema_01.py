from numpy import array, linspace, log10
from matplotlib.cbook import flatten
import matplotlib.pyplot as plt


def Kullback_Leibler_distance(theta1: float, theta2: float) -> array:
    value = (1-theta1)*log10((1-theta1)/(1-theta2))
    value += theta1*log10((theta1)/(theta2))
    return value


parameters = {"path graphics": "../Graphics/",
              "file graphics": "Kullback.png",
              "theta 1": linspace(0.1, 0.9, 9),
              "theta 2": linspace(0.001, 0.999, 50)}
fig, axs = plt.subplots(3, 3,
                        sharex=True,
                        sharey=True,
                        figsize=(9, 9),
                        )
axs = list(flatten(axs))
colors = ["#f94144",
          "#f3722c",
          "#f8961e",
          "#f9844a",
          "#f9c74f",
          "#90be6d",
          "#43aa8b",
          "#4d908e",
          "#577590",
          "#277da1"]
for theta1, color, ax in zip(parameters["theta 1"], colors, axs):
    distance = Kullback_Leibler_distance(theta1,
                                         parameters["theta 2"])
    ax.set_title("$\\theta_1$={:.1f}".format(theta1))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 2.5)
    ax.grid(ls="--",
            color="#000000",
            alpha=0.5)
    ax.plot(parameters["theta 2"],
            distance,
            color=color,
            lw=3)
fig.text(0.5, 0.01,
         '$\\theta_2$',
         ha='center',
         va='center')
fig.text(0.015, 0.5,
         "Kullback Leibler distance",
         ha='center',
         va='center',
         rotation='vertical')
plt.tight_layout(pad=2)
plt.savefig("{}{}".format(parameters["path graphics"],
                          parameters["file graphics"]))
