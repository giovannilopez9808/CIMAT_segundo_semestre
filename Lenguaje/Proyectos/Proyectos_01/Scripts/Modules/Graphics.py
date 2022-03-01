from .functions import join_path, obtain_name_place_from_filename
from .datasets import parameters_model
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from pandas import DataFrame
from numpy import arange


def plot_bars(data: DataFrame, dataset: parameters_model, parameters: dict) -> None:
    """
    Función que realiza el ploteo de barras, una grafica por todas los lugares
    """
    # Obtiene el total de lugares
    n_bar = len(parameters["keys"])
    # Ancho de las barras
    width = parameters["width"]
    # Numero de lugares en números consecutivos
    values = arange(len(data.index))
    # Nombres de los lugares con saltos de linea
    names = [name.replace(" ", "\n") for name in data.index]
    fig, ax = plt.subplots(figsize=(12, 4))
    bars = []
    # Ploteo de las barras
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
    # Numero de cada barra
    for bar in bars:
        ax.bar_label(bar,
                     padding=1,
                     fmt=parameters["format"],
                     )
    # Ticks de las grafcias
    ax.set_xticks(values, names)
    ax.set_ylim(0, parameters["y lim"])
    ax.set_yticks(arange(0,
                         parameters["y lim"] + parameters["y delta"],
                         parameters["y delta"]))
    ax.grid(ls="--",
            color="#000000",
            alpha=0.5,
            axis="y")
    # Nombre de guardado
    filename = join_path(dataset.parameters["path graphics"],
                         parameters["file graphics"])
    # Leyenda
    fig.legend(frameon=False,
               ncol=n_bar,
               bbox_to_anchor=(0.6, 1))
    plt.tight_layout(pad=2)
    # Guardado de la grafica
    plt.savefig(filename, dpi=300)


def plot_ages_histogram(ages: list, dataset: parameters_model, parameters: dict) -> None:
    """
    Ploteo de los histogramas por edades
    """
    # Obtiene el nombre del lugar
    nameplace = obtain_name_place_from_filename(parameters["filename"])
    # Obtiene el nombre de guardado
    filename = join_path(dataset.parameters["path graphics"],
                         parameters["filename"])
    #  Edad minima y maxima
    min_age = ages.min()
    max_age = ages.max()
    # Definicion de los limites
    ylim = dataset.graphics[nameplace]["y lim"]
    ydelta = dataset.graphics[nameplace]["y delta"]
    # Lista de edades consecutivas
    ages_list = arange(min_age, max_age, 1, dtype=int)
    bins = int(max_age - min_age)
    fig, ax = plt.subplots(figsize=(14, 6))
    # Histograma
    ax.hist(ages,
            bins=bins,
            align="left",
            color="#9d0208",
            alpha=0.5,
            edgecolor='white')
    # Limites de la grafica
    ax.set_xlim(min_age, max_age + 1)
    ax.set_xticks(ages_list)
    ax.set_ylim(0, ylim)
    ax.set_yticks(arange(0, ylim + ydelta, ydelta))
    ax.set_xlabel("Edades")
    ax.set_ylabel("Número de usuarios")
    ax.grid(ls="--", alpha=0.5, color="#000000", axis="y")
    plt.tight_layout()
    # Guardado de la grafica
    plt.savefig(filename, dpi=300)


def plot_word_cloud(data: DataFrame, dataset: parameters_model, parameters: dict, show: bool = False) -> None:
    wc = WordCloud()
    plt.subplots(figsize=(8.5, 4))
    wc.generate_from_frequencies(data)
    plt.axis("off")
    plt.imshow(wc, interpolation="bilinear")
    plt.tight_layout()
    if show:
        plt.show()
    else:
        filename = join_path(dataset.parameters["path graphics"],
                             parameters["wordcloud name"])
        plt.savefig(filename,
                    bbox_inches="tight",
                    pad_inches=0,)
