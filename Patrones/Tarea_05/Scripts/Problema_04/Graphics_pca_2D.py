from Modules.params import get_params
import matplotlib.pyplot as plt
from pandas import read_csv
from os.path import join

params = get_params()
params["file results"] = "PCA.csv"
params["file graphics"] = "PCA_2D.png"
data_types = [["Component 1", "Component 2"],
              ["Component 2", "Component 3"],
              ["Component 1", "Component 3"]]
filename = join(params["path results"],
                params["file results"])
data = read_csv(filename)
fig, axs = plt.subplots(1, 3,
                        figsize=(16, 6))
for i, (label, color) in enumerate(params["type cash"].items()):
    subset = data[data["label"] == i]
    for j, data_type in enumerate(data_types):
        ax = axs[j]
        data_type1, data_type2 = data_type
        ax.scatter(subset[data_type1],
                   subset[data_type2],
                   color=color,
                   label=label,
                   alpha=0.5,
                   )
        ax.set_xlabel(data_type1)
        ax.set_ylabel(data_type2)
        ax.set_xticks([])
        ax.set_yticks([])
handles, labels = ax.get_legend_handles_labels()
fig.legend(handles,
           labels,
           frameon=False,
           ncol=2,
           fontsize=16,
           loc='upper center',
           )
fig.text(0.025,
         0.94,
         "A",
         fontsize=14)
fig.text(0.35,
         0.94,
         "B",
         fontsize=14)
fig.text(0.68,
         0.94,
         "C",
         fontsize=14)
plt.tight_layout(pad=3)
filename = join(params["path graphics"],
                params["file graphics"])
plt.savefig(filename,
            dpi=300)
