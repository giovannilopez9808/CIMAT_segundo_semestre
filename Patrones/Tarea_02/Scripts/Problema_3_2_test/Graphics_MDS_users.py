import matplotlib.pyplot as plt
from functions import *

parameters = obtain_parameters()
parameters["file MDS"] = "Test/MDS_users.csv"
parameters["file graphics"] = "Test/MDS_users.png"
movies = read_movie_list()
movies = split_genres(movies)
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
