from Modules.datasets import parameters_model
from Modules.functions import join_path
from Modules.Graphics import plot_bars
from pandas import read_csv

parameters = {"file results": "mean_std_scores.csv",
              "file graphics": "mean_std_scores.png",
              "width": 0.45,
              "y lim": 5,
              "y delta": 0.5,
              "keys": ["Mean", "std"],
              "labels": ["Media", "Desviación estandar"],
              "format": "%.3f"}
dataset = parameters_model()
filename = join_path(dataset.parameters["path results"],
                     parameters["file results"])
data = read_csv(filename, index_col=0)
data = data.transpose()
plot_bars(data, dataset, parameters)
