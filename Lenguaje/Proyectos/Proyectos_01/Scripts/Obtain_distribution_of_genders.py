from Modules.functions import join_path, obtain_name_place_from_filename, ls
from Modules.datasets import parameters_model
from Modules.tripadvisor import tripadvisor_model
from pandas import DataFrame

filename = "distribution_genders.csv"
dataset = parameters_model()
tripadvisor = tripadvisor_model(dataset)
files = ls(dataset.parameters["path data"])
results = {}
result_basis = {"Masculino": 0,
                "Femenino": 0}
for file in files:
    nameplace = obtain_name_place_from_filename(file)
    tripadvisor.read_data(file)
    result = tripadvisor.data["Género"].value_counts()
    results[nameplace] = result_basis.copy()
    for key in result.keys():
        results[nameplace][key] = result[key]
results = DataFrame(results)
results.index.name = "Gender"
filename = join_path(dataset.parameters["path results"],
                     filename)
results.to_csv(filename)
