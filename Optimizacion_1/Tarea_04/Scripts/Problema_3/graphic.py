import matplotlib.pyplot as plt
from os import listdir as ls
from datasets import *
from auxiliar import *

datasets = graphics_datasets_class()
parameters = datasets.obtain_dataset("lambda 1000")
files = ls(parameters["path data"])
graphics_name = obtain_graphics_filename(parameters)
file = obtain_filename(parameters)
colors = obtain_colors_per_method()
plt.subplots(figsize=(8, 5))
filename = join_path(parameters["path data"],
                     file)
data = read_data(filename)
plt.xlim(parameters["xlim"][0],
         parameters["xlim"][1])
plt.ylim(parameters["y lim"][0],
         parameters["y lim"][1])
plt.plot(data["t"],
         data["x"],
         label="$x(t)$",
         color=colors["x"],)
plt.plot(data["t"],
         data["y"],
         label="$y(t)$",
         color=colors["y"],)
plt.grid(ls="--",
         alpha=0.5,
         color="#000000")
plt.ylabel("$f$")
plt.xlabel("t")
plt.xticks(parameters["xticks"])
plt.legend(frameon=False,
           ncol=2,
           bbox_to_anchor=(0.5, 1.08))
plt.tight_layout(pad=1)
# plt.show()
plt.savefig(join_path(parameters["path graphics"], graphics_name), dpi=400)
