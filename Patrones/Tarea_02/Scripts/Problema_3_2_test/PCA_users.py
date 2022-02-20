from sklearn.decomposition import PCA
from functions import *
parameters = obtain_parameters()
parameters["file data"] = "Test/ratings_select_movies.csv"
parameters["file results"] = "Test/PCA_users.csv"
data = np.loadtxt(join_path(parameters["path results"],
                            parameters["file data"]),
                  delimiter=",",
                  skiprows=1)
users_id = data[:, 0]
data = np.delete(data, 0, axis=1)
results = pd.DataFrame(index=users_id,
                       columns=["x", "y"])
pca = PCA()
data_model = pca.fit_transform(data)
results["x"] = data_model[:, 0]
results["y"] = data_model[:, 1]
results.to_csv(join_path(parameters["path results"],
                         parameters["file results"]),
               index_label="users_id")
