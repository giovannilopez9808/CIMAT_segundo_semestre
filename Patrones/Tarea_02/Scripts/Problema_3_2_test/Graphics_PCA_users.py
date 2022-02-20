import matplotlib.pyplot as plt
from functions import *

parameters = obtain_parameters()
parameters["file MDS"] = "Test/PCA_users.csv"
parameters["file graphics"] = "Test/PCA_users.png"
movies = read_movie_list()
movies = split_genres(movies)
genres = obtain_select_genres(only_names=False)
data = read_data(parameters["path results"],
                 parameters["file MDS"],
                 use_index=True)
plt.scatter(data["x"],
            data["y"],
            alpha=0.3)
plt.axis("off")
plt.tight_layout()
plt.savefig(join_path(parameters["path graphics"],
                      parameters["file graphics"]))
