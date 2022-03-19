from Modules.functions import join_path, ls
from Modules.dataset import obtain_parms
import matplotlib.pyplot as plt
from pandas import read_csv
from numpy import linspace

dataset = {"k-means++": {"color": "#f72585",
                         "ls": "-"},
           "random": {"color": "#3a0ca3",
                      "ls": "--"},
           "array": {"color": "#4cc9f0",
                     "ls": "--"}}
params = obtain_parms()
files = ls(params["path results"])
plt.subplots(figsize=(8, 4))
for file in files:
    filename = join_path(params["path results"],
                         file)
    init = file.replace(".csv", "")
    init = init.replace("_", "-")
    data = read_csv(filename,
                    index_col=0)
    minimum = data.min()
    data_len = len(minimum)
    plt.plot(range(data_len),
             minimum,
             label=init,
             alpha=0.5,
             ls=dataset[init]["ls"],
             marker=".",
             color=dataset[init]["color"])
plt.legend(frameon=False,
           ncol=3)
plt.xlim(0, data_len-1)
plt.xticks(range(data_len), data.columns)
plt.ylim(0, 12000)
plt.yticks(linspace(0, 12000, 13))
plt.ylabel("score")
plt.xlabel("clusters")
plt.grid(ls="--",
         color="#000000",
         alpha=0.5)
plt.tight_layout()
filename = join_path(params["path graphics"],
                     "minimum_scores.png")
plt.savefig(filename,
            dpi=300)
