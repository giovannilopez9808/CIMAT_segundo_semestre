from Modules.functions import obtain_parameters
from Modules.models import LLE_model_class
from Modules.dataset import data_class
from Modules.graphics import plot

parameters = obtain_parameters()
parameters["file graphics"] = "LLE.png"
data = data_class(parameters)
model = LLE_model_class()
model.run(data.matrix)
plot(model,
     data.names,
     parameters)
