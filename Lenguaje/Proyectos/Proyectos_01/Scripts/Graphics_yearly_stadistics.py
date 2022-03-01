from Modules.datasets import parameters_model
from Modules.functions import join_path, ls
import matplotlib.pyplot as plt
from pandas import read_csv

datasets = parameters_model()
datasets.parameters["path results"] += "Yearly/"
datasets.parameters["path graphics"] += "Yearly/"
files = ls(datasets.parameters["path results"])
for file in files:
    filename = join_path(datasets.parameters["path results"],
                         file)
    data = read_csv(filename,
                    index_col=0,
                    parse_dates=[0])
    fig, (ax1, ax2) = plt.subplots(2, 1,
                                   figsize=(12, 4),
                                   sharex=True)
    ax1.plot(data.index,
             data["0"],
             marker="o",
             label="Negativos",
             color="#80b918")
    ax1.plot(data.index,
             data["1"],
             marker="o",
             label="Neutro",
             color="#52b69a")
    ax1.plot(data.index,
             data["2"],
             marker="o",
             label="Positivo",
             color="#1e6091")
    ax2.errorbar(data.index,
                 data["Escala mean"],
                 data["Escala std"],
                 fmt='o',
                 markersize=8,
                 capsize=10,
                 color="#9d0208",
                 label="$\\mu$ y $\\sigma$ scores")
    ax1.set_ylim(0, 1)
    ax1.grid(ls="--",
             color="#000000",
             alpha=0.5)
    ax2.set_xlabel("Años")
    ax1.set_ylabel("Frecuencia relativa (%)")
    ax2.set_xlim(data.index[0], data.index[-1])
    ax2.set_xticks(data.index)
    ax2.set_xticklabels([date.year for date in data.index])
    ax2.set_ylabel("Score")
    ax2.set_ylim(0, 6)
    ax2.set_yticks([value for value in range(7)])
    ax2.grid(ls="--",
             color="#000000",
             alpha=0.5)
    fig.legend(frameon=False,
               ncol=4,
               bbox_to_anchor=(0.7, 1))
    plt.tight_layout(pad=2)
    filename = join_path(datasets.parameters["path graphics"],
                         file.replace(".csv", ".png"))
    plt.savefig(filename, dpi=300)
