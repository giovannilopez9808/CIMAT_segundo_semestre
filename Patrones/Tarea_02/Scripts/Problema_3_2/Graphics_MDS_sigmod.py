from functions import *

parameters = obtain_parameters()
parameters["file data"] = "MDS_sigmod.csv"
parameters["file graphics"] = "MDS_sigmod.png"
plot_MDS_results(parameters)
