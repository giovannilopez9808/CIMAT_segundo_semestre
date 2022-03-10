from Modules.models import Isomap_model_class, TSNE_model_class, LLE_model_class, SOM_model_class
from Modules.functions import obtain_parameters
from Modules.dataset import data_class
from Modules.graphics import plot

parameters = obtain_parameters()
parameters["file graphics"] = "SOM_LLE.png"
data = data_class(parameters)
SOM_model = SOM_model_class()
SOM_model.run(data.matrix)
SOM_model.create_classes_dataframe(data.names)
print(SOM_model.classes)
LLE_model = LLE_model_class()
LLE_model.run(data.matrix)
plot(LLE_model,
     data.names,
     parameters,
     include_images=False,
     alpha=1,
     color=SOM_model.results)

parameters["file graphics"] = "SOM_TSNE.png"
TSNE_model = TSNE_model_class()
TSNE_model.run(data.matrix)
plot(TSNE_model,
     data.names,
     parameters,
     include_images=False,
     alpha=1,
     color=SOM_model.results)

parameters["file graphics"] = "SOM_isomap.png"
Isomap_model = Isomap_model_class()
Isomap_model.run(data.matrix)
plot(Isomap_model,
     data.names,
     parameters,
     include_images=False,
     alpha=1,
     color=SOM_model.results)
