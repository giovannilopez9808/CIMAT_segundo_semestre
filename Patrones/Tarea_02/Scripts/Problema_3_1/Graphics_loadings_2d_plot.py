from functions import *

parameters = {"path data": "../../Results/Problema_3_1/",
              "file loadings": "pca_components.csv",
              "path graphics": "../../Graphics/Problema_3_1/",
              "file graphics": "loadings2d.png"}

data = read_data(parameters["path data"],
                 parameters["file loadings"],
                 use_index=True)
data = data.transpose()
sets = obtain_sets()
plt.scatter(data["PC1"],
            data["PC2"],
            marker=".",
            c="#34a0a4",
            alpha=0.7)
for i, set in enumerate(sets):
    data_set = data[data.index == str(set)]
    plt.text(data_set["PC1"]+0.01,
             data_set["PC2"]+0.01,
             str(i+1),
             fontsize=14,
             color="#003049")
    plt.scatter(data_set["PC1"],
                data_set["PC2"],
                marker="o",
                c="#007200")
plt.xlim(0.1, 0.7)
plt.xticks(np.linspace(0.1, 0.7, 7))
plt.ylim(-0.1, 1)
plt.yticks(np.linspace(-0.1, 1, 12))
plt.xlabel("1er componente")
plt.ylabel("2da componente")
plt.grid(ls="--")
plt.tight_layout()
plt.savefig("{}{}".format(parameters["path graphics"],
                          parameters["file graphics"]),
            dpi=400)
