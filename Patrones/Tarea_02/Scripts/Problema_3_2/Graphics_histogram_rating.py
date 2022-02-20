import matplotlib.pyplot as plt
from functions import *

parameters = obtain_parameters()
colors = obtain_colors()
parameters["graphics file"] = "Histogram_rating.png"
ratings = read_rating_list()
labels = obtain_all_scores()
yticks = np.linspace(0, 30000, 13)
height, values, hist_list = plt.hist(ratings["rating"],
                                     bins=10,
                                     alpha=0.7,
                                     rwidth=0.85)
for x, y, label in zip(values, height, labels):
    plt.text(x+0.125, y+100, "{}".format(label))
for color, i in zip(colors, range(len(hist_list))):
    hist_list[i].set_facecolor(color)
plt.xticks([])
plt.yticks(yticks)
plt.ylim(0, 30000)
plt.ylabel("Conteo de calificación")
plt.grid(ls="--",
         color="#000000",
         alpha=0.7)
plt.tight_layout()
plt.savefig(join_path(parameters["path graphics"],
                      parameters["graphics file"]))
