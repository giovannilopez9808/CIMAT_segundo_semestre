from Modules.functions import obtain_parameters
from Modules.graphics import plot_denograma
from Modules.dataset import data_class

parameters = obtain_parameters()
parameters["file graphics"] = "denograms.png"
data = data_class(parameters)
plot_denograma(parameters,
               data)
