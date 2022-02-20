from functions import *

parameters = obtain_parameters()
parameters["file data"] = "MDS_euclidean.csv"
parameters["file graphics"] = "MDS_euclidean.png"
plot_MDS_results(parameters)
