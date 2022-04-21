from Modules.params import get_params
import matplotlib.pyplot as plt
from pandas import read_csv
from os.path import join

params = get_params()
params["file results"] = "PCA.csv"
params["file graphics"] = "PCA_3D.png"
filename = join(params["path results"],
                params["file results"])
data = read_csv(filename)
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(projection='3d')
for i, (label, color) in enumerate(params["type cash"].items()):
    subset = data[data["label"] == i]
    ax.scatter(subset["Component 1"],
               subset["Component 2"],
               subset["Component 3"],
               alpha=0.5,
               label=label,
               color=color)
ax.set_xlabel("Component 1")
ax.set_ylabel("Component 2")
ax.set_zlabel("Component 3")
ax.view_init(22, 155)
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
