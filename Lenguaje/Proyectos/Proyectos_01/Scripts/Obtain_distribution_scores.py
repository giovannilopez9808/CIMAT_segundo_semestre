from Modules.functions import join_path, obtain_name_place_from_filename
from Modules.datasets import parameters_model
from Modules.tripadvisor import tripadvisor_model
from pandas import DataFrame
from os import listdir as ls

filename = "distribution_scores.csv"
dataset = parameters_model()
tripadvisor = tripadvisor_model(dataset)
files = sorted(ls(dataset.parameters["path data"]))
results = {}
result_basis = {"Positivo": 0,
                "Neutro": 0,
                "Negativo": 0}
for file in files:
    nameplace = obtain_name_place_from_filename(file)
    tripadvisor.read_data(file)
    result = tripadvisor.data["Escala"].value_counts()
    results[nameplace] = result_basis.copy()
    for key in result.keys():
        if key in [4, 5]:
            key_name = "Positivo"
        if key in [3]:
            key_name = "Neutro"
        if key in [1, 2]:
            key_name = "Negativo"
        results[nameplace][key_name] = result[key]
results = DataFrame(results)
results.index.name = "Calificación"
filename = join_path(dataset.parameters["path results"],
                     filename)
results.to_csv(filename)
