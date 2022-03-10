from Modules.models import TSNE_model_class, Isomap_model_class, LLE_model_class, cluster_model_class
from Modules.functions import obtain_parameters
from Modules.graphics import plot_clusters
from Modules.dataset import data_class

models_set = {"LLE": {"model": LLE_model_class(),
                      "file graphics": "Cluster_LLE.png"},
              "Isomap": {"model": TSNE_model_class(),
                         "file graphics": "Cluster_isomap.png"},
              "TSNE": {"model": Isomap_model_class(),
                       "file graphics": "Cluster_TSNE.png"}}

cluster_model = cluster_model_class()
parameters = obtain_parameters()
data = data_class(parameters)

for model_name in models_set:
    model_set = models_set[model_name]
    parameters["file graphics"] = model_set["file graphics"]
    model = model_set["model"]
    model.run(data.matrix)
    plot_clusters(parameters,
                  cluster_model,
                  model,
                  data)
