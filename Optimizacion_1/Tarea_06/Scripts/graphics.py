from Modules.functions import join_path, obtain_filename
from Modules.datasets import datasets_class
import matplotlib.pyplot as plt
from pandas import read_csv
from numpy import linspace

datasets = datasets_class()
parameters = datasets.obtain_dataset("log likelihood bisection")
filename = obtain_filename(parameters)
parameters["file graphics"] = filename.replace(".csv", ".png")
filename = join_path(parameters["path results"], filename)
data = read_csv(filename)
fig, ((ax1, ax3), (ax2, ax4)) = plt.subplots(2, 2, figsize=(8, 5))
ax1.plot(data.index, data["fx"], color="#1b3a4b")
ax2.plot(data.index, data["dfx"], color="#9d0208")
ax1.set_xlim(0, 500)
ax1.set_yticks(linspace(0, 600, 7))
ax1.set_ylabel("$f\;(x)$")
ax1.grid(ls="--", color="#000000", alpha=0.5)
ax1.set_ylim(0, 600)
ax2.set_xlim(0, 500)
ax2.set_ylim(0, 300)
ax2.set_yticks(linspace(0, 500, 6))
ax2.set_xlabel("Iteraciones")
ax2.set_ylabel("$\\nabla\;f(x)$")
ax2.grid(ls="--", color="#000000", alpha=0.5)

ax3.plot(data.index, data["fx"], color="#1b3a4b")
ax3.set_xlim(0, 50)
ax3.set_ylim(0, 600)
ax3.set_ylabel("$f\;(x)$")
ax3.grid(ls="--", color="#000000", alpha=0.5)

ax4.plot(data.index, data["dfx"], color="#9d0208")
ax4.set_xlim(0, 50)
ax4.set_ylim(0, 500)
ax4.set_xlabel("Iteraciones")
ax4.set_ylabel("$\\nabla\;f(x)$")
ax4.grid(ls="--", color="#000000", alpha=0.5)
plt.tight_layout()
plt.savefig(join_path(parameters["path graphics"],
                      parameters["file graphics"]),
            dpi=400)
