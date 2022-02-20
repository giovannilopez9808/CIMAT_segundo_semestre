from sklearn.decomposition import PCA
from functions import *
parameters = obtain_parameters()
parameters["file data"] = "Test/ratings_select_movies.csv"
parameters["file results"] = "Test/PCA_movies.csv"
data = np.loadtxt(join_path(parameters["path results"],
                            parameters["file data"]),
                  delimiter=",",
                  skiprows=1)
data = np.transpose(data)
data = np.delete(data, 0, axis=0)
movie_id = np.loadtxt(join_path(parameters["path results"],
                                parameters["file data"]),
                      delimiter=",",
                      max_rows=1,
                      dtype="str")
movie_id = np.delete(movie_id, 0)
results = pd.DataFrame(index=movie_id,
                       columns=["x", "y"])
pca = PCA()
data_model = pca.fit_transform(data)
results["x"] = data_model[:, 0]
results["y"] = data_model[:, 1]
results.to_csv(join_path(parameters["path results"],
                         parameters["file results"]),
               index_label="movie_id")
