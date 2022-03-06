from fcntl import F_SETFL
from .functions import obtain_stopwords
from .functions import join_path
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from numpy import array
import warnings


def create_constellation(reduced_matrix: array, target_words: array, parameters: dict, show: bool = False) -> None:
    warnings.filterwarnings("ignore")
    stopwords = obtain_stopwords("spanish")
    plt.figure(figsize=(40, 40), dpi=100)
    plt.scatter(reduced_matrix[:, 0],
                reduced_matrix[:, 1],
                20,
                color='black')
    for idx, word in enumerate(target_words[:]):
        x = reduced_matrix[idx, 0]
        y = reduced_matrix[idx, 1]
        if word in stopwords:
            plt.annotate(word, (x, y), color='red')
        else:
            plt.annotate(word, (x, y), color='black')
    plt.tight_layout()
    plt.axis("off")
    if show:
        plt.show()
    else:
        filename = join_path(parameters["path graphics"],
                             parameters["constellation name"])
        plt.savefig(filename)


def create_centroid(reduced_matrix: array, target_words: array, parameters: dict, show: bool = False) -> None:
    fig, ax = plt.subplots(figsize=(15, 15))
    for idx, word in enumerate(target_words[:]):
        if word in parameters["subset words"]:
            position = reduced_matrix[idx]
            ax.scatter(position[0],
                       position[1],
                       c="#000000")
            ax.arrow(0, 0,
                     position[0],
                     position[1],
                     head_width=0.8,
                     head_length=0.8,
                     fc='r',
                     ec='r',
                     width=1e-2)
            ax.annotate(word,
                        position)
    ax.axis("off")
    plt.tight_layout()
    if show:
        plt.show()
    else:
        filename = join_path(parameters["path graphics"],
                             parameters["centroid name"])
        plt.savefig(filename)


def create_wordcloud(data: dict, parameters: dict, show: bool = False) -> None:
    wc = WordCloud()
    wc.generate_from_frequencies(data)
    plt.axis("off")
    plt.imshow(wc, interpolation="bilinear")
    if show:
        plt.show()
    else:
        filename = join_path(parameters["path graphics"],
                             parameters["wordcloud name"])
        plt.savefig(filename)
