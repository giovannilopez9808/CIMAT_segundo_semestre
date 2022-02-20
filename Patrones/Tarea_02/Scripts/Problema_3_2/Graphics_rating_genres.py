from functions import *

parameters = obtain_parameters()
parameters["file result"] = "Rating_genre.csv"
parameters["file graphics"] = "Rating_genre.png"
genres = obtain_all_genres(only_names=True)
genres[14] = "No listed"
scores = obtain_all_scores()
data = np.loadtxt(join_path(parameters["path results"],
                            parameters["file result"]),
                  skiprows=1,
                  usecols=np.arange(1, 21),
                  delimiter=",")
data = data+1
data = np.log10(data)
plt.subplots(figsize=(12, 7))
plt.imshow(data,
           aspect="auto",
           cmap="plasma",
           origin="lower")
colorbar = plt.colorbar()
colorbar.set_label("Calificación en escala logarítmica",
                   rotation=-90,
                   fontsize=13,
                   labelpad=25)
plt.xticks(np.arange(len(genres)),
           genres,
           rotation=60)
plt.ylabel("Calificaciones")
plt.yticks(np.arange(len(scores)),
           scores)
plt.tight_layout()
plt.savefig(join_path(parameters["path graphics"],
                      parameters["file graphics"]))
