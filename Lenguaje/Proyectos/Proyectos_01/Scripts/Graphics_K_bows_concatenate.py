from Modules.datasets import parameters_model
from Modules.Graphics import plot_word_cloud
from Modules.functions import join_path, ls
from pandas import read_csv


parameters = {"wordcloud name": ""}
dataset = parameters_model()
dataset.parameters["path graphics"] += "BoW_concatenate/"
dataset.parameters["path results"] += "BoW_concatenate/"
files = ls(dataset.parameters["path results"])
for file in files:
    parameters["wordcloud name"] = file.replace(".csv", ".png")
    filename = join_path(dataset.parameters["path results"],
                         file)
    data = read_csv(filename,
                    index_col=0)
    data = data.iloc[:50]
    plot_word_cloud(data["Scores"].to_dict(),
                    dataset,
                    parameters)
