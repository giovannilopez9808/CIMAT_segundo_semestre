from Modules.functions import join_path, obtain_name_place_from_filename
from Modules.datasets import parameters_model
from Modules.tripadvisor import tripadvisor_model
from pandas import DataFrame
from os import listdir as ls

filename = "distribution_nationality.csv"
dataset = parameters_model()
tripadvisor = tripadvisor_model(dataset)
files = sorted(ls(dataset.parameters["path data"]))
results = {}
result_basis = {"Nacional": 0,
                "Internacional": 0}
for file in files:
    nameplace = obtain_name_place_from_filename(file)
    tripadvisor.read_data(file)
    result = tripadvisor.data["Nacional ó Internacional"].value_counts()
    results[nameplace] = result_basis.copy()
    for key in result.keys():
        results[nameplace][key] = result[key]
results = DataFrame(results)
results.index.name = "Nationality"
filename = join_path(dataset.parameters["path results"],
                     filename)
results.to_csv(filename)
