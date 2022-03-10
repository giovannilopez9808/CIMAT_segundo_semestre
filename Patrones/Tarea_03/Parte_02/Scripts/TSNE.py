from Modules.functions import obtain_parameters
from Modules.models import TSNE_model_class
from Modules.dataset import data_class
from Modules.graphics import plot

parameters = obtain_parameters()
parameters["file graphics"] = "TSNE.png"
data = data_class(parameters)
model = TSNE_model_class(2)
model.run(data.matrix)
plot(model,
     data.names,
     parameters)
