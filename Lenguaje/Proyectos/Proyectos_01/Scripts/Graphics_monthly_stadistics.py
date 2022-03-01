from Modules.datasets import parameters_model
from Modules.functions import join_path, ls
from pandas import read_csv, to_datetime
import matplotlib.pyplot as plt

datasets = parameters_model()
datasets.parameters["path results"] += "Monthly/"
datasets.parameters["path graphics"] += "Monthly/"
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
    years = sorted(set([date.year for date in data.index]))
    dates = list(to_datetime(["{}-01-01".format(year) for year in years]))
    dates += [to_datetime("{}-01-01".format(dates[-1].year+1))]
    years += [dates[-1].year]
    ax1.scatter(data.index,
                data["0"],
                marker=".",
                label="Negativos",
                color="#80b918")
    ax1.scatter(data.index,
                data["1"],
                marker=".",
                label="Neutro",
                color="#52b69a")
    ax1.scatter(data.index,
                data["2"],
                marker=".",
                label="Positivo",
                color="#1e6091")
    ax2.fill_between(data.index,
                     data["Escala mean"]-data["Escala std"],
                     data["Escala mean"]+data["Escala std"],
                     label="$\\sigma$ scores",
                     color="#ff8500",
                     alpha=0.5)
    ax2.scatter(data.index,
                data["Escala mean"],
                marker='.',
                #  markersize=5,
                #  capsize=7,
                color="#3c096c",
                label="$\\mu$ scores")
    ax1.set_ylim(0, 1)
    ax1.grid(ls="--",
             color="#000000",
             alpha=0.5)
    ax2.set_xlabel("Años")
    ax1.set_ylabel("Frecuencia relativa (%)")
    ax2.set_xlim(dates[0], to_datetime(dates[-1]))
    ax2.set_xticks(dates)
    ax2.set_xticklabels(years)
    ax2.set_ylabel("Score")
    ax2.set_ylim(0, 6)
    ax2.set_yticks([value for value in range(7)])
    ax2.grid(ls="--",
             color="#000000",
             alpha=0.5)
    fig.legend(frameon=False,
               ncol=5,
               bbox_to_anchor=(0.75, 1))
    plt.tight_layout(pad=2)
    filename = join_path(datasets.parameters["path graphics"],
                         file.replace(".csv", ".png"))
    plt.savefig(filename, dpi=300)
