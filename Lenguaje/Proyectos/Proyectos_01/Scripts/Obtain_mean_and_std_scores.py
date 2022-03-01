from Modules.functions import join_path, obtain_name_place_from_filename, ls
from Modules.datasets import parameters_model
from Modules.tripadvisor import tripadvisor_model
from pandas import DataFrame

filename = "mean_std_scores.csv"
dataset = parameters_model()
tripadvisor = tripadvisor_model(dataset)
files = ls(dataset.parameters["path data"])
results = {}
result_basis = {"Mean": 0.0,
                "std": 0.0}
for file in files:
    nameplace = obtain_name_place_from_filename(file)
    tripadvisor.read_data(file)
    mean = tripadvisor.data["Escala"].mean()
    std = tripadvisor.data["Escala"].std()
    results[nameplace] = result_basis.copy()
    results[nameplace]["Mean"] = mean
    results[nameplace]["std"] = std
results = DataFrame(results)
results.index.name = "Results"
filename = join_path(dataset.parameters["path results"],
                     filename)
results.to_csv(filename)
