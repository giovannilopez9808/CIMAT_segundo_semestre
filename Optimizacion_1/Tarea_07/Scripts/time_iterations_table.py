from Modules.params import get_datasets, get_params, obtain_filename_iteration, obtain_path, join
from tabulate import tabulate
from pandas import DataFrame, read_csv


params = get_params()
datasets = get_datasets()
columns = ["Time", "Function", "Iterations"]
for column in columns:
    print("-"*50)
    print("\t\t", column)
    print("-"*50)
    table = []
    header = ["Funcion"]
    header += [step_model for step_model in datasets["step models"]]
    for function in datasets["functions"]:
        data_function = [function]
        for step_model in datasets["step models"]:
            dataset = {
                "function": function,
                "step model": step_model,
            }
            path = obtain_path(params, dataset)
            filename = join(path,
                            params["file stadistics"])
            data = read_csv(filename)
            mean = data[column].mean()
            data_function += [mean]
        table += [data_function]
    print(tabulate(table,
                   headers=header))
