from sklearn.manifold import Isomap
import matplotlib.pyplot as plt
from numpy import array, dot
from pandas import DataFrame
from functions import *


def plot_isomap(genres: list, data: DataFrame, matrix: array, fig: plt.figure,
                neighbors: int, position: int):
    isomap = Isomap(n_neighbors=neighbors, n_components=3)
    isomap.fit(matrix)
    results = isomap.transform(matrix)
    results = DataFrame(results,
                        columns=['Component 1', 'Component 2', "Component 3"])
    data["Component 1"] = list(results["Component 1"])
    data["Component 2"] = list(results["Component 2"])
    data["Component 3"] = list(results["Component 3"])
    ax = fig.add_subplot(position, projection='3d')
    ax.set_title("Vecinos {}".format(neighbors))
    for genre in genres:
        data_genre = data[data["genres"] == genre]
        if len(data_genre):
            ax.scatter(data_genre["Component 1"],
                       data_genre["Component 2"],
                       data_genre["Component 3"],
                       c=data_genre["colors"],
                       marker=".",
                       label=genre)
    ax.view_init(27, 112)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    return ax


parameters = obtain_parameters()
parameters["file data"] = "select_rating.csv"
data, rating = obtain_vector_of_rating(parameters)
rating = array(rating)
movies = read_movie_list()
genres = obtain_select_genres()
movies = split_genres(movies)
data = assign_genre(movies, data)
matrix = dot(rating.reshape(-1, 1), rating.reshape(1, -1))
fig = plt.figure(figsize=(8, 8))
ax = plot_isomap(genres, data, matrix, fig, 2, 221)
ax = plot_isomap(genres, data, matrix, fig, 4, 222)
ax = plot_isomap(genres, data, matrix, fig, 6, 223)
ax = plot_isomap(genres, data, matrix, fig, 8, 224)
handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', ncol=2, frameon=False)
plt.savefig("test.png", dpi=300)
