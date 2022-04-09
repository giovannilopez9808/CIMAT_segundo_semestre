from Modules.params import get_datasets, get_params, obtain_path
from tabulate import tabulate
from pandas import read_csv
from os.path import join

params = get_params()
datasets = get_datasets()
columns = ["Time",
           "Function",
           "Iterations"]
header = ["Funcion"]
header += [step_model.replace("-", " ")
           for step_model in datasets["step models"]]
for column in columns:
    print("-"*80)
    print("\t\t\t\t", column)
    print("-"*80)
    table = []
    for function in datasets["functions"]:
        data_function = [function]
        for step_model in datasets["step models"]:
            dataset = {
                "function": function,
                "step model": step_model,
            }
            path = obtain_path(params,
                               dataset)
            filename = join(path,
                            params["file stadistics"])
            data = read_csv(filename)
            mean = data[column].mean()
            data_function += [mean]
        table += [data_function]
    print(tabulate(table, headers=header))
