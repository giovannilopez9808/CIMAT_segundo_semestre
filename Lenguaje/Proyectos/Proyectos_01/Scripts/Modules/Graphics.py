from .functions import join_path, obtain_name_place_from_filename
from .datasets import parameters_model
import matplotlib.pyplot as plt
from pandas import DataFrame
from numpy import arange


def plot_bars(data: DataFrame, dataset: parameters_model, parameters: dict) -> None:
    n_bar = len(parameters["keys"])
    width = parameters["width"]
    values = arange(len(data.index))
    names = [name.replace(" ", "\n") for name in data.index]
    fig, ax = plt.subplots(figsize=(12, 4))
    bars = []
    for i, (key, label, color) in enumerate(zip(parameters["keys"],
                                                parameters["labels"],
                                                parameters["colors"])):
        bar = ax.bar(values + width*(n_bar*i-n_bar)/n_bar,
                     data[key],
                     width=width,
                     label=label,
                     edgecolor='white',
                     color=color)
        bars += [bar]
    for bar in bars:
        ax.bar_label(bar,
                     padding=1,
                     fmt=parameters["format"],
                     )
    ax.set_xticks(values, names)
    ax.set_ylim(0, parameters["y lim"])
    ax.set_yticks(arange(0,
                         parameters["y lim"] + parameters["y delta"],
                         parameters["y delta"]))
    ax.grid(ls="--",
            color="#000000",
            alpha=0.5,
            axis="y")
    filename = join_path(dataset.parameters["path graphics"],
                         parameters["file graphics"])
    fig.legend(frameon=False,
               ncol=n_bar,
               bbox_to_anchor=(0.6, 1))
    plt.tight_layout(pad=2)
    plt.savefig(filename, dpi=300)


def plot_ages_histogram(ages: list, dataset: parameters_model,
                        parameters: dict) -> None:
    nameplace = obtain_name_place_from_filename(parameters["filename"])
    filename = join_path(dataset.parameters["path graphics"],
                         parameters["filename"])
    min_age = ages.min()
    max_age = ages.max()
    ylim = dataset.graphics[nameplace]["y lim"]
    ydelta = dataset.graphics[nameplace]["y delta"]
    ages_list = arange(min_age, max_age, 1, dtype=int)
    bins = int(max_age - min_age)
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.hist(ages,
            bins=bins,
            align="left",
            color="#9d0208",
            alpha=0.5,
            edgecolor='white')
    ax.set_xlim(min_age, max_age + 1)
    ax.set_xticks(ages_list)
    ax.set_ylim(0, ylim)
    ax.set_yticks(arange(0, ylim + ydelta, ydelta))
    ax.set_xlabel("Edades")
    ax.set_ylabel("Número de usuarios")
    ax.grid(ls="--", alpha=0.5, color="#000000", axis="y")
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
