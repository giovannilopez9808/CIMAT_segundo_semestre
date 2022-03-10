from Modules.functions import obtain_parameters
from Modules.models import Isomap_model_class
from Modules.dataset import data_class
from Modules.graphics import plot

parameters = obtain_parameters()
parameters["file graphics"] = "Isomap.png"
data = data_class(parameters)
model = Isomap_model_class()
model.run(data.matrix)
plot(model,
     data.names,
     parameters)
