from sklearn.manifold import MDS
from functions import *


def euclidean(vector: np.array) -> np.array:
    n = len(vector)
    matrix = np.zeros((n, n))
    for i in range(n):
        v_i = vector[i]
        for j in range(i, n):
            v_j = vector[j]
            matrix[i, j] += (v_i-v_j)**2
            matrix[j, i] += (v_i-v_j)**2
    return matrix


parameters = obtain_parameters()
parameters["file data"] = "select_rating.csv"
parameters["file results"] = "MDS_euclidean.csv"
data = read_data(parameters["path results"],
                 parameters["file data"], use_index=True)
vector = list(data["rating"])
matrix = euclidean(vector)
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
