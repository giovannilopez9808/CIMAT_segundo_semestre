from sklearn.manifold import MDS
from functions import *
parameters = obtain_parameters()
parameters["file data"] = "Test/ratings_select_movies.csv"
parameters["file results"] = "Test/MDS_users.csv"
data = np.loadtxt(join_path(parameters["path results"],
                            parameters["file data"]),
                  delimiter=",",
                  skiprows=1)
users_id = data[:, 0]
data = np.delete(data, 0, axis=1)
results = pd.DataFrame(index=users_id,
                       columns=["x", "y"])
model2d = MDS(random_state=42)
data_model = model2d.fit(data)
results_matrix = model2d.embedding_
results["x"] = results_matrix[:, 0]
results["y"] = results_matrix[:, 1]
results.to_csv(join_path(parameters["path results"],
                         parameters["file results"]),
               index_label="users_id")
