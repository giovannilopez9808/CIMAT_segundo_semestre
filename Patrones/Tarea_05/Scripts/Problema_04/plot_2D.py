from Modules.dataset import dataset_model, join
from Modules.params import get_params
import matplotlib.pyplot as plt

params = get_params()
params["file graphics"] = "plot_2D.png"
dataset = dataset_model(params)
fig, axs = plt.subplots(1, 3,
                        figsize=(16, 6))
for label, color in params["type cash"].items():
    dataset.select_cash_type(label)
    for ax, data_types in zip(axs, params["pair plots"]):
        data_type1, data_type2 = data_types
        ax.set_ylim(0, 20)
        ax.set_xlabel(data_type1,
                      fontsize=16)
        ax.set_ylabel(data_type2,
                      fontsize=16)
        data_1 = dataset.data_select[data_type1]
        data_2 = dataset.data_select[data_type2]
        ax.scatter(data_1,
                   data_2,
                   alpha=0.7,
                   label=label,
                   color=color,
                   )
handles, labels = ax.get_legend_handles_labels()
fig.legend(handles,
           labels,
           frameon=False,
           ncol=2,
           fontsize=16,
           loc='upper center',
           )
plt.tight_layout(pad=3)
filename = join(params["path graphics"],
                params["file graphics"])
plt.savefig(filename,
            dpi=300)
