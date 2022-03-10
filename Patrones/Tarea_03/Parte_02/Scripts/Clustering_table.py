from Modules.models import LLE_model_class, cluster_model_class
from Modules.functions import obtain_parameters
from Modules.dataset import data_class
from pandas import DataFrame

cluster_model = cluster_model_class()
parameters = obtain_parameters()
data = data_class(parameters)
model = LLE_model_class()
model.run(data.matrix)
results = DataFrame()
for linkage in parameters["linkage types"]:
    cluster_model.run(data.matrix, 4, linkage)
    cluster_model.create_classes_dataframe(data.names)
    results[linkage] = cluster_model.classes["Classes"]
results.to_csv("../Results/cluster_labels.csv")
