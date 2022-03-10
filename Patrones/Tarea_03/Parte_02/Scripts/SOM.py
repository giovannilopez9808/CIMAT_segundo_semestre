from Modules.models import Isomap_model_class, TSNE_model_class, LLE_model_class, SOM_model_class
from Modules.functions import obtain_parameters
from Modules.dataset import data_class
from Modules.graphics import plot

models_set = {"LLE": {"model": LLE_model_class(),
                      "file graphics": "SOM_LLE.png"},
              "Isomap": {"model": TSNE_model_class(),
                         "file graphics": "SOM_isomap.png"},
              "TSNE": {"model": Isomap_model_class(),
                       "file graphics": "SOM_TSNE.png"}}
parameters = obtain_parameters()
data = data_class(parameters)
SOM_model = SOM_model_class()
SOM_model.run(data.matrix)
SOM_model.create_classes_dataframe(data.names)
SOM_model.classes.to_csv("../Results/SOM_labels.csv")

for model_name in models_set:
    model_set = models_set[model_name]
    parameters["file graphics"] = model_set["file graphics"]
    model = model_set["model"]
    model.run(data.matrix)
    plot(model,
         data.names,
         parameters,
         include_images=False,
         alpha=1,
         color=SOM_model.results)
