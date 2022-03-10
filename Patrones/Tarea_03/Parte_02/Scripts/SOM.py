from Modules.models import LLE_model_class, SOM_model_class
from Modules.functions import obtain_parameters
from Modules.dataset import data_class
from Modules.graphics import plot

parameters = obtain_parameters()
parameters["file graphics"] = "SOM.png"
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
