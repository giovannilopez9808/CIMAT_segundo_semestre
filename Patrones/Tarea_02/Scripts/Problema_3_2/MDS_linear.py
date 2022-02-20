from sklearn.manifold import MDS
from functions import *

parameters = obtain_parameters()
parameters["file data"] = "select_rating.csv"
parameters["file results"] = "MDS_linear.csv"
data = read_data(parameters["path results"],
                 parameters["file data"],
                 use_index=True)
vector = list(data["rating"])
matrix = linear(vector)
results = pd.DataFrame(index=data.index,
                       columns=["x", "y"])
model2d = MDS(random_state=42,
              dissimilarity='precomputed')
data_model = model2d.fit_transform(matrix)
results_matrix = model2d.embedding_
results["x"] = results_matrix[:, 0]
results["y"] = results_matrix[:, 1]
results.to_csv(join_path(parameters["path results"],
                         parameters["file results"]),
               index_label="rating")
