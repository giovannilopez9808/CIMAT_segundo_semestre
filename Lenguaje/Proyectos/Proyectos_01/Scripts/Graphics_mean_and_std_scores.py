from Modules.datasets import parameters_model
from Modules.functions import join_path
from pandas import DataFrame, read_csv
import matplotlib.pyplot as plt
from numpy import arange


def plot_bars(data: DataFrame, dataset: parameters_model, parameters: dict) -> None:
    mean = data["Mean"]
    std = data["std"]
    width = parameters["width"]
    values = arange(len(data.index))
    names = [name.replace(" ", "\n") for name in data.index]
    fig, ax = plt.subplots(figsize=(12, 4))
    bar1 = ax.bar(values-width/2,
                  mean,
                  width=width,
                  label="Media",
                  color="#144552")
    bar2 = ax.bar(values+width/2,
                  std,
                  width=width,
                  label="Desviación estandar",
                  color="#4d194d")
    ax.bar_label(bar1, padding=1, fmt="%.3f")
    ax.bar_label(bar2, padding=1, fmt="%.3f")
    ax.set_xticks(values, names)
    ax.set_ylim(0, 5)
    ax.set_yticks(arange(0, 5.5, 0.5))
    ax.grid(ls="--",
            color="#000000",
            alpha=0.5,
            axis="y")
    filename = join_path(dataset.parameters["path graphics"],
                         parameters["file graphics"])
    fig.legend(frameon=False,
               ncol=2,
               bbox_to_anchor=(0.6, 1))
    plt.tight_layout(pad=2)
    plt.savefig(filename, dpi=300)


parameters = {"file results": "mean_std_scores.csv",
              "file graphics": "mean_std_scores.png",
              "width": 0.45}
dataset = parameters_model()
filename = join_path(dataset.parameters["path results"],
                     parameters["file results"])
data = read_csv(filename, index_col=0)
data = data.transpose()
plot_bars(data, dataset, parameters)
