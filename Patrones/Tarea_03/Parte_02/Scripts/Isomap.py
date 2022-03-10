from Modules.functions import join_path, obtain_parameters, plot_image
from Modules.models import Isomap_model_class
from Modules.dataset import data_class
import matplotlib.pyplot as plt

parameters = obtain_parameters()
parameters["file graphics"] = "Isomap.png"
data = data_class(parameters)
Isomap_model = Isomap_model_class(3, 2)
Isomap_model.run(data.matrix)
print(Isomap_model.results)
fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(Isomap_model.results["Component 1"],
           Isomap_model.results["Component 2"],
           alpha=0.5,
           color="#370617")
for index in Isomap_model.animals_index:
    animal_image = "{}.jpg".format(data.names[index])
    position = [Isomap_model.results["Component 1"][index],
                Isomap_model.results["Component 2"][index]]
    plot_image(parameters["path graphics animales"],
               animal_image,
               ax,
               position
               )
plt.axis("off")
plt.tight_layout()
plt.savefig(join_path(parameters["path graphics"],
                      parameters["file graphics"]))
