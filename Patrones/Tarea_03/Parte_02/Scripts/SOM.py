from Modules.functions import join_path, obtain_parameters, plot_image
from Modules.models import SOM_model_class
from Modules.dataset import data_class
import matplotlib.pyplot as plt

parameters = obtain_parameters()
parameters["file graphics"] = "SOM.png"
data = data_class(parameters)
SOM_model = SOM_model_class()
SOM_model.run(data.matrix)
SOM_model.create_classes_dataframe(data.names)
print(SOM_model.classes)
fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(data.matrix[:, 0],
           data.matrix[:, 1],
           c=SOM_model.results)
plt.axis("off")
plt.tight_layout()
plt.savefig(join_path(parameters["path graphics"],
                      parameters["file graphics"]))
