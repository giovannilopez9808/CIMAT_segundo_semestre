from Modules.datasets import parameters_model
from Modules.functions import join_path
from Modules.Graphics import plot_bars
from pandas import read_csv

parameters = {"file results": "distribution_nationality.csv",
              "file graphics": "distribution_nationality.png",
              "width": 0.45,
              "y lim": 1000,
              "y delta": 100,
              "keys": ["Nacional", "Internacional"],
              "labels": ["Nacional", "Internacional"],
              "format": "%.0f"}
dataset = parameters_model()
filename = join_path(dataset.parameters["path results"],
                     parameters["file results"])
data = read_csv(filename, index_col=0)
data = data.transpose()
plot_bars(data, dataset, parameters)
