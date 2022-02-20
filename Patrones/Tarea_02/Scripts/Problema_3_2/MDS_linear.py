from functions import *

parameters = obtain_parameters()
parameters["file data"] = "select_rating.csv"
parameters["file results"] = "MDS_linear.csv"
data, vector = obtain_vector_of_rating(parameters)
matrix = linear(vector)
run_MDS_model(data, matrix, parameters)
