from sklearn.manifold import MDS
from functions import *

parameters = obtain_parameters()
parameters["file data"] = "select_rating.csv"
parameters["file results"] = "MDS_euclidean.csv"
data, vector = obtain_vector_of_rating(parameters)
matrix = euclidean(vector)
run_MDS_model(data, matrix, parameters)
