from functions import *

parameters = obtain_parameters()
parameters["file data"] = "select_rating.csv"
parameters["file results"] = "MDS_gaussian.csv"
data, vector = obtain_vector_of_rating(parameters)
matrix = gaussian(vector, 0.5)
run_MDS_model(data, matrix, parameters)
