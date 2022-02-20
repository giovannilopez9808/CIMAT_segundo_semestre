from functions import *

parameters = obtain_parameters()
parameters["file data"] = "MDS_linear.csv"
parameters["file graphics"] = "MDS_linear.png"
plot_MDS_results(parameters)
