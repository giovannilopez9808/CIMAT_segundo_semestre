from Modules.models import TSNE_model_class, cluster_model_class
from Modules.functions import obtain_parameters
from Modules.graphics import plot_clusters
from Modules.dataset import data_class

cluster_model = cluster_model_class()
parameters = obtain_parameters()
parameters["file graphics"] = "Cluster_TSNE.png"
data = data_class(parameters)
model = TSNE_model_class()
model.run(data.matrix)
plot_clusters(parameters,
              cluster_model,
              model,
              data)