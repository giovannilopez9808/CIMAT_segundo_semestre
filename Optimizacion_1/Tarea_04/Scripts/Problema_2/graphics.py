from datasets_graphics import *
import matplotlib.pyplot as plt
from os import listdir as ls
from auxiliar import *

datasets = graphics_datasets_class()
parameters = datasets.obtain_dataset("rosembrock 2 predefined")
files = ls(parameters["path data"])
graphics_name = obtain_graphics_filename(parameters)
files = obtain_files_per_dataset(files,
                                 parameters)
colors = obtain_colors_per_method()
fig, (ax1, ax2) = plt.subplots(2, 1,
                               sharex=True,
                               figsize=(6, 5))
for file in files:
    method = obtain_method_from_name(file)
    color = colors[method]
    filename = join_path(parameters["path data"],
                         file)
    data = read_data(filename)
    ax1.set_xlim(parameters["xlim"][0],
                 parameters["xlim"][1])
    ax1.set_ylim(parameters["y1 lim"][0],
                 parameters["y1 lim"][1])
    ax2.set_ylim(parameters["y2 lim"][0],
                 parameters["y2 lim"][1])
    ax1.plot(data.index,
             data["fx"],
             label=method,
             color=color,
             marker=".",
             ls="--")
    ax2.plot(data.index,
             data["dfx"],
             label=method,
             color=color,
             marker=".",
             ls="--")
    ax1.grid(ls="--",
             alpha=0.5,
             color="#000000")
    ax2.grid(ls="--",
             alpha=0.5,
             color="#000000")
    ax1.set_ylabel("$f\;(X)$")
    ax2.set_ylabel("$\\nabla f\;(X)$")
    ax2.set_xlabel("Iteraciones")
    ax2.set_xticks(parameters["xticks"])
handles, labels = ax1.get_legend_handles_labels()
fig.legend(frameon=False,
           handles=handles,
           labels=labels,
           ncol=2,
           bbox_to_anchor=(0.8, 1.01))
plt.tight_layout(pad=2)
plt.savefig(join_path(parameters["path graphics"],
                      graphics_name),
            dpi=400)
