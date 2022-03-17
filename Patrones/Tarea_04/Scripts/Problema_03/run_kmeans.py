from Modules.normal_data import datasets_model
from Modules.datasets import obtain_params

params = obtain_params()
datasets = datasets_model()
datasets.run_kmeans()
datasets.run_kmeans_with_validation()
datasets.save(params)
