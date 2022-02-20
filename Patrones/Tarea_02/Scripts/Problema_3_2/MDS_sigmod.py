from functions import *

parameters = obtain_parameters()
parameters["file data"] = "select_rating.csv"
parameters["file results"] = "MDS_sigmod.csv"
data, vector = obtain_vector_of_rating(parameters)
matrix = sigmod(vector, 0.75, 0.2)
run_MDS_model(data, matrix, parameters)
