from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from functions import *

parameters = {"path data": "../../Results/Problema_3_1/",
              "file loadings": "pca_components.csv",
              "path graphics": "../../Graphics/Problema_3_1/",
              "file graphics": "loadings_3d.png"}

data = read_data(parameters["path data"],
                 parameters["file loadings"],
                 use_index=True)
data = data.transpose()
sets = obtain_sets_3d()
fig = plt.figure(figsize=(8, 7))
ax = fig.add_subplot(projection='3d')
ax.scatter(data["PC1"],
           data["PC2"],
           data["PC3"],
           marker=".",
           c="#34a0a4",
           alpha=0.3)
for i, set in enumerate(sets):
    data_set = data[data.index == str(set)]
    ax.scatter(data_set["PC1"],
               data_set["PC2"],
               data_set["PC3"],
               marker="o",
               s=20,
               c=sets[set],
               label="Imagen {}".format(i+1))
ax.set_xlim(0.1, 0.7)
ax.set_ylim(-0.1, 1)
ax.set_xlabel("1er componente")
ax.set_ylabel("2da componente")
ax.set_zlabel("3ra componente")
ax.grid(ls="--")
ax.view_init(3, -44)
fig.legend(ncol=3,
           frameon=False,
           bbox_to_anchor=(0.9, 0.83))
plt.tight_layout()
print(data)
plt.savefig("{}{}".format(parameters["path graphics"],
                          parameters["file graphics"]),
            dpi=400)
