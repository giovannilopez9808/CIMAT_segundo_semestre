import matplotlib.pyplot as plt
from functions import *

parameters = obtain_parameters()
parameters["file data"] = "MDS_gaussian.csv"
parameters["file graphics"] = "MDS_gaussian.png"
movies = read_movie_list()
genres = obtain_select_genres()
movies = split_genres(movies)
data = read_data(parameters["path results"],
                 parameters["file data"],
                 use_index=True)
data = assign_genre(movies, data)
for genre in genres:
    data_genre = data[data["genres"] == genre]
    if len(data_genre):
        plt.scatter(data_genre["x"],
                    data_genre["y"],
                    c=data_genre["colors"],
                    label=genre,
                    alpha=0.2)
plt.axis("off")
plt.legend(ncol=2)
plt.tight_layout()
plt.savefig(join_path(parameters["path graphics"],
                      parameters["file graphics"]))
