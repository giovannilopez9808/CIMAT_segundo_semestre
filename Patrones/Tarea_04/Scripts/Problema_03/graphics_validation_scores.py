from Modules.datasets import obtain_params
from Modules.functions import join_path
import matplotlib.pyplot as plt
from pandas import read_csv
from numpy import linspace


params = obtain_params()
params["file scores"] = "validation_scores.csv"
params["file graphics"] = "validation_scores.png"
scores = read_csv(join_path(params["path results"],
                            params["file scores"]),
                  index_col=0)
plt.subplots(figsize=(8, 5))
plt.plot(scores,
         marker="o",
         ls="--",
         color="#5e60ce")
plt.xlim(2, 15)
plt.ylim(0, 700)
plt.xticks(linspace(2, 15, 14))
plt.yticks(linspace(0, 700, 15))
plt.ylabel("Función objetivo")
plt.xlabel("Clusters")
plt.grid(ls="--",
         color="#000000",
         alpha=0.5)
plt.tight_layout()
plt.savefig(join_path(params["path graphics"],
                      params["file graphics"]),
            dpi=300)
