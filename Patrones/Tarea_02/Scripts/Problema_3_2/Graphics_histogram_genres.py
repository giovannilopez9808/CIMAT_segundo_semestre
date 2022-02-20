import matplotlib.pyplot as plt
from functions import *

parameters = obtain_parameters()
parameters["graphics file"] = "Histogram_genres.png"
parameters["file results"] = "Histogram_genres.csv"
colors = obtain_colors()
data = read_data(parameters["path results"],
                 parameters["file results"])
data.columns = ["Genre", "Count"]
data.loc[len(data)-1, "Genre"] = "No listed"
yticks = np.linspace(0, 44000, 12)
plt.subplots(figsize=(12, 8))
bars = plt.bar(data.index, data["Count"])
for index in data.index:
    value = data["Count"][index]
    genre = data["Genre"][index]
    plt.text(index-0.4,
             value+100,
             "{}".format(value))
for color, i in zip(colors, range(len(bars))):
    bars[i].set_facecolor(color)
plt.xticks(data.index,
           data["Genre"],
           rotation=60)
plt.yticks(yticks)
plt.ylim(0, 44000)
plt.ylabel("Conteo de calificación")
plt.grid(axis="y",
         ls="--",
         color="#000000",
         alpha=0.7)
plt.tight_layout()
plt.savefig(join_path(parameters["path graphics"],
                      parameters["graphics file"]))
