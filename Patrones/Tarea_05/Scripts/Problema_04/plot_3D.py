from Modules.dataset import dataset_model, join
from Modules.params import get_params
import matplotlib.pyplot as plt

params = get_params()
params["file graphics"] = "plot_3D.png"
dataset = dataset_model(params)
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(projection='3d')
for label, color in params["type cash"].items():
    dataset.select_cash_type(label)
    ax.scatter(dataset.data_select["variance"],
               dataset.data_select["skewness"],
               dataset.data_select["curtosis"],
               alpha=0.5,
               label=label,
               color=color)
    ax.set_xlabel("variance")
    ax.set_ylabel("skewness")
    ax.set_zlabel("curtosis")
ax.view_init(9, -54)
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False
ax.xaxis.pane.set_edgecolor('#ffffff')
ax.yaxis.pane.set_edgecolor('#ffffff')
ax.zaxis.pane.set_edgecolor('#ffffff')
plt.legend(frameon=False,
           ncol=2,
           loc='upper center',
           )
plt.tight_layout()
filename = join(params["path graphics"],
                params["file graphics"])
plt.savefig(filename,
            dpi=300)
