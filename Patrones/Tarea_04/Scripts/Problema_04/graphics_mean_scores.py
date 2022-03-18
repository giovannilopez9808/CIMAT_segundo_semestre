from Modules.dataset import obtain_parms
from Modules.functions import join_path
import matplotlib.pyplot as plt
from pandas import read_csv
from numpy import linspace

params = obtain_parms()
filename = join_path(params["path results"],
                     "kmean.csv")
data = read_csv(filename,
                index_col=0)
mean = data.mean()
minimum = data.min()
data_len = len(mean)
plt.subplots(figsize=(8, 4))
plt.plot(range(data_len),
         mean,
         ls="--",
         color="#f72585",
         marker=".",
         label="mean")
plt.plot(range(data_len),
         minimum,
         alpha=0.5,
         marker=".",
         color="#4361ee",
         label="minimum")
plt.legend(frameon=False,
           ncol=2)
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
                     "scores.png")
plt.savefig(filename)
