from Modules.Graphics import plot_monthly_results
from Modules.datasets import parameters_model
from Modules.functions import join_path, ls
from pandas import read_csv

datasets = parameters_model()
datasets.parameters["path results"] += "Monthly/"
datasets.parameters["path graphics"] += "Monthly/"
files = ls(datasets.parameters["path results"])
parameters = {"file graphics": ""}
for file in files:
    parameters["file graphics"] = file.replace(".csv", ".png")
    filename = join_path(datasets.parameters["path results"],
                         file)
    data = read_csv(filename,
                    index_col=0,
                    parse_dates=[0])
    plot_monthly_results(data, datasets, parameters)
