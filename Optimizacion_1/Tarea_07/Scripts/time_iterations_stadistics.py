from Modules.params import get_datasets, get_params, obtain_filename_iteration, obtain_path, join
from pandas import DataFrame, read_csv


params = get_params()
datasets = get_datasets()
for function in datasets["functions"]:
    for step_model in datasets["step models"]:
        results = DataFrame(columns=["Function",
                                     "Iterations",
                                     "Time"])
        dataset = {
            "function": function,
            "step model": step_model,
        }
        for iteration in range(datasets["iteration"]):
            filename = obtain_filename_iteration(params,
                                                 dataset,
                                                 iteration+1)
            data = read_csv(filename,
                            index_col=0)
            n = len(data)-1
            results.loc[iteration] = [data["Function"][n],
                                      n,
                                      data["Time"][n]
                                      ]
        path = obtain_path(params, dataset)
        filename = join(path,
                        params["file stadistics"])
        results.to_csv(filename,
                       index=False)
