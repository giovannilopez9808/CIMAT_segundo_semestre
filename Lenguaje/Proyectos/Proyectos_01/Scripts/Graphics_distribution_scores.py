from Modules.datasets import parameters_model
from Modules.functions import join_path
from Modules.Graphics import plot_bars
from pandas import read_csv

parameters = {"file results": "distribution_scores.csv",
              "file graphics": "distribution_scores.png",
              "width": 0.35,
              "y lim": 600,
              "y delta": 100,
              "keys": ["Positivo", "Neutro", "Negativo"],
              "labels": ["Positivo", "Neutro", "Negativo"],
              "colors":   ["#d9ed92", "#34a0a4", "#184e77"],
              "format": "%.0f"}
dataset = parameters_model()
filename = join_path(dataset.parameters["path results"],
                     parameters["file results"])
data = read_csv(filename, index_col=0)
data = data.transpose()
plot_bars(data, dataset, parameters)
