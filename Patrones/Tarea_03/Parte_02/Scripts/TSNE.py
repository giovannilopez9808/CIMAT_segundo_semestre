from Modules.functions import join_path, obtain_parameters, plot_image
from Modules.models import TSNE_model_class
from Modules.dataset import data_class
import matplotlib.pyplot as plt

parameters = obtain_parameters()
parameters["file graphics"] = "TSNE.png"
data = data_class(parameters)
LLE_model = TSNE_model_class(2)
LLE_model.run(data.matrix)
fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(LLE_model.results["x"],
           LLE_model.results["y"],
           alpha=0.5,
           color="#370617")
for index in LLE_model.animals_index:
    animal_image = "{}.jpg".format(data.names[index])
    position = [LLE_model.results["x"][index],
                LLE_model.results["y"][index]]
    plot_image(parameters["path graphics animales"],
               animal_image,
               ax,
               position
               )
plt.axis("off")
plt.tight_layout()
plt.savefig(join_path(parameters["path graphics"],
                      parameters["file graphics"]))
