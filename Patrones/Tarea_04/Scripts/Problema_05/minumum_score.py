from Modules.dataset import obtain_parms
from Modules.functions import join_path
import matplotlib.pyplot as plt
from pandas import read_csv


params = obtain_parms()
filename = join_path(params["path results"],
                     "scores.csv")
data = read_csv(filename,
                index_col=0)
minumum = data.min()
plt.subplots(figsize=(9, 4))
plt.plot(params["k values"],
         minumum,
         ls="--",
         marker="o",
         color="#6a040f")
plt.xlim(2, 6)
plt.xticks(params["k values"])
plt.ylim(0, 1.4e14)
plt.xlabel("Cluster")
plt.ylabel("Score")
plt.grid(ls="--",
         color="#000000",
         alpha=0.5)
plt.tight_layout()
filename = join_path(params["path graphics"],
                     "minumum_score.png")
plt.savefig(filename)
