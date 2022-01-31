import matplotlib.pyplot as plt
import pandas as pd

parameters = {"path data": "../../Data/",
              "path graphics": "../../Document/Graphics/",
              "data file": "heptatlon.csv",
              "graphics file": "heptatlon.png",
              "colors": ["f94144",
                         "0466c8",
                         "90be6d",
                         "001233",
                         "f9844a",
                         "02c39a",
                         "43aa8b",
                         "4d908e",
                         "577590",
                         "277da1"]
              }

data = pd.read_csv("{}{}".format(parameters["path data"],
                                 parameters["data file"]),
                   index_col=0)
names = data.index
names = [name.split(" ")[0].capitalize() for name in names]
data.index = names
data = data.drop(columns="score")
mean = data.mean()
std = data.std()
fig, (ax1, ax2) = plt.subplots(2,
                               1,
                               figsize=(10, 5),
                               sharex=True)
for i in range(7):
    column = data.columns[i]
    color = "#{}".format(parameters["colors"][i])
    if i % 2 == 0:
        ax = ax1
    else:
        ax = ax2
    data_column = (data[column] - mean[column])/std[column]
    ax.plot(data_column,
            color=color,
            label=column,
            lw=2,
            marker="o",
            ls="--",)
for ax in [ax1, ax2]:
    ax.set_ylim(-3, 3)
    ax.grid()
fig.legend(ncol=7,
           frameon=False,
           loc="upper center",
           bbox_to_anchor=[0.5, 1.015])
plt.xticks(rotation=30)
fig.tight_layout()
plt.savefig("{}{}".format(parameters["path graphics"],
                          parameters["graphics file"]))
plt.clf()
