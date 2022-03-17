from sklearn.metrics import calinski_harabasz_score
from Modules.datasets import obtain_params
from numpy import array, linspace, loadtxt
from Modules.functions import join_path
from pandas import DataFrame, read_csv
import matplotlib.pyplot as plt


def obtain_calinski_harabasz_score(data: array, labels: DataFrame) -> array:
    ch_scores = [calinski_harabasz_score(data, labels[k])
                 for k in labels]
    return ch_scores


params = obtain_params()
params["file train labels"] = "train_labels.csv"
params["file train data"] = "train_data.csv"
params["file validation labels"] = "validation_labels.csv"
params["file validation data"] = "validation_data.csv"
params["file graphics"] = "calinski_harabasz_score.png"
train_data = loadtxt(join_path(params["path results"],
                               params["file train data"]),
                     delimiter=",")
train_labels = read_csv(join_path(params["path results"],
                                  params["file train labels"]),
                        index_col=0)
validation_data = loadtxt(join_path(params["path results"],
                                    params["file validation data"]),
                          delimiter=",")
validation_labels = read_csv(join_path(params["path results"],
                                       params["file validation labels"]),
                             index_col=0)
train_ch_scores = obtain_calinski_harabasz_score(train_data,
                                                 train_labels)
validation_ch_scores = obtain_calinski_harabasz_score(validation_data,
                                                      validation_labels)
plt.subplots(figsize=(8, 5))
plt.plot(range(2, 16),
         train_ch_scores,
         marker="o",
         ls="--",
         color="#52b69a",
         label="Train")
plt.plot(range(2, 16),
         validation_ch_scores,
         marker="o",
         ls="--",
         color="#5e60ce",
         label="Validation")
plt.xlim(2, 15)
plt.ylim(0, 900)
plt.xticks(linspace(2, 15, 14))
plt.yticks(linspace(0, 900, 10))
plt.ylabel("Calinski Harabasz score")
plt.xlabel("Clusters")
plt.grid(ls="--",
         color="#000000",
         alpha=0.5)
plt.legend(frameon=False,
           ncol=2,
           fontsize=12)
plt.tight_layout()
plt.savefig(join_path(params["path graphics"],
                      params["file graphics"]),
            dpi=300)
