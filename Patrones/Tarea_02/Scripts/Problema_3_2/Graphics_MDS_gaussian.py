from functions import *

parameters = obtain_parameters()
parameters["file data"] = "MDS_gaussian.csv"
parameters["file graphics"] = "MDS_gaussian.png"
plot_MDS_results(parameters)
