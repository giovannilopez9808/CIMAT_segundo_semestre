import matplotlib.pyplot as plt
from functions import *

parameters = obtain_parameters()
parameters["file MDS"] = "Test/PCA_movies.csv"
parameters["file graphics"] = "Test/PCA_movies.png"
movies = read_movie_list()
movies = split_genres(movies)
genres = obtain_select_genres(only_names=False)
data = read_data(parameters["path results"],
                 parameters["file MDS"],
                 use_index=True)
data["genre"] = ""
for movie_id in data.index:
    genres_list = movies["genres"][movie_id]
    for genre in genres_list:
        if genre in genres:
            data.loc[movie_id, "genre"] = genre
            break
for genre in genres:
    data_genre = data[data["genre"] == genre]
    plt.scatter(data_genre["x"],
                data_genre["y"],
                c=genres[genre],
                label=genre,
                alpha=0.3)
plt.legend(frameon=False,
           ncol=3,
           bbox_to_anchor=(0.6, 1.1))
plt.axis("off")
plt.tight_layout()
plt.savefig(join_path(parameters["path graphics"],
                      parameters["file graphics"]))
