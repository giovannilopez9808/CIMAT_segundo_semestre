import matplotlib.pyplot as plt
import pandas as pd

parameters = {"path data": "../../Data/",
              "path graphics": "../../Document/Graphics/",
              "data file": "heptatlon.csv"}

data = pd.read_csv("{}{}".format(parameters["path data"],
                                 parameters["data file"]),
                   index_col=0)
names = data.index
names = [name.split(" ")[0].capitalize() for name in names]
data.index = names
data = data.drop(columns="score")
for column in data:
    filename = column.replace(" ", "")
    data_column = data[column]
    plt.subplots(figsize=(10, 5))
    plt.subplots_adjust(left=0.061,
                        bottom=0.148,
                        right=0.972,
                        top=0.975)
    plt.plot(data_column,
             color="#d62828",
             lw=2,
             marker="o",
             ls="--"
             )
    plt.xticks(rotation=30)
    plt.savefig("{}{}".format(parameters["path graphics"],
                              filename))
    plt.clf()
