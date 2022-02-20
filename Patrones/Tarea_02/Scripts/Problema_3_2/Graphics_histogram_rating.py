import matplotlib.pyplot as plt
from functions import *

parameters = obtain_parameters()
colors = obtain_colors()
parameters["graphics file"] = "Histogram_rating.png"
ratings = read_rating_list()
ratings = list(ratings["rating"])
labels = obtain_all_scores()
yticks = np.linspace(0, 30000, 13)
set_rating = set(ratings)
data = pd.DataFrame(0,
                    index=set_rating,
                    columns=["Count"])
for rating in set_rating:
    data["Count"][rating] = ratings.count(rating)
bars = plt.bar(data.index,
               list(data["Count"]),
               alpha=0.7,
               width=0.4
               )
for x, y in zip(data.index, data["Count"]):
    plt.text(x-0.2, y+100, "{:.0f}".format(y))
for color, i in zip(colors, range(len(data))):
    bars[i].set_facecolor(color)
plt.xlabel("Calificaciones")
plt.xticks(data.index)
plt.yticks(yticks
           )
plt.ylim(0, 30000)
plt.ylabel("Conteo de calificación")
plt.grid(ls="--",
         color="#000000",
         alpha=0.7,
         axis="y")
plt.tight_layout()
plt.savefig(join_path(parameters["path graphics"],
                      parameters["graphics file"]))
