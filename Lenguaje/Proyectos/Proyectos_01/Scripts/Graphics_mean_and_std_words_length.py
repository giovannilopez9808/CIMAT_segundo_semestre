from Modules.datasets import parameters_model
from Modules.functions import join_path
from Modules.Graphics import plot_bars
from pandas import read_csv

parameters = {"file results": "mean_std_words_length.csv",
              "file graphics": "mean_std_words_length.png",
              "width": 0.45,
              "y lim": 70,
              "y delta": 10}
dataset = parameters_model()
filename = join_path(dataset.parameters["path results"],
                     parameters["file results"])
data = read_csv(filename, index_col=0)
data = data.transpose()
plot_bars(data, dataset, parameters)
