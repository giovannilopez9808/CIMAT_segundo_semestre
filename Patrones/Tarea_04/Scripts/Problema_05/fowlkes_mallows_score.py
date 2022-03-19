from sklearn.metrics import fowlkes_mallows_score
from Modules.dataset import obtain_parms
from Modules.functions import join_path
from pandas import DataFrame, read_csv
import matplotlib.pyplot as plt
from numpy import array


def obtain_fm_score(true_label: array, label: array) -> float:
    score = fowlkes_mallows_score(true_label,
                                  label)
    return score


params = obtain_parms()
results = {}
for k in params["k values"]:
    results[k] = []
    filename = "cluster_{}.csv".format(k)
    filename = join_path(params["path results"],
                         filename)
    data = read_csv(filename,
                    index_col=0)
    true_label = array(data["True"])
    data = data.drop(columns="True")
    for column in data:
        label = array(data[column])
        score = obtain_fm_score(true_label,
                                label)
        results[k] += [score]
results = DataFrame(results)
mean = results.mean()
filename = join_path(params["path graphics"],
                     "fowlkes_mallows_score.png")
plt.subplots(figsize=(9, 4))
plt.plot(params["k values"],
         mean,
         ls="--",
         color="#52b69a",
         marker="o")
plt.xticks(params["k values"])
plt.xlim(2, 6)
plt.xlabel("Cluster")
plt.ylabel("Fowlkes Mallows Score")
plt.ylim(0.4, 0.75)
plt.grid(ls="--",
         color="#000000",
         alpha=0.5)
plt.tight_layout()
plt.savefig(filename,
            dpi=300)
