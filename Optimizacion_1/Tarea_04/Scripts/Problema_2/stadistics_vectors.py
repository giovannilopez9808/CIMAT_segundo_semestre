from datasets import datasets_statistics_class
from auxiliar import join_path, read_data

datasets = datasets_statistics_class()
for dataset in datasets.datasets:
    print("-"*40)
    print(dataset)
    parameters = datasets.obtain_dataset(dataset)
    filename = join_path(parameters["path results"],
                         parameters["filename"])
    data = read_data(filename)
    results = data.describe()
    results = results.drop(["25%", "50%", "75%"])
    results.columns = ["f(x)", "||nabla f(x)||"]
    results.index = ["n", "Media", "sigma", "Mínimo", "Máximo"]
    print(results)
