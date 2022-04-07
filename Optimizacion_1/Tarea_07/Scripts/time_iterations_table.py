from Modules.params import get_datasets, get_params, obtain_filename_iteration, obtain_path, join
from pandas import DataFrame, read_csv


params = get_params()
datasets = get_datasets()
for function in datasets["functions"]:
    for step_model in datasets["step models"]:
        dataset = {
            "function": function,
            "step model": step_model,
        }
        path = obtain_path(params, dataset)
        filename = join(path,
                        params["file stadistics"])
        data = read_csv(filename)
        mean = data.mean()
        print(mean)
