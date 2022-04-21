from Modules.dataset import dataset_model, join
from Modules.params import get_params
from Modules.pca import PCA_model
from numpy import mean, std

params = get_params()
pca = PCA_model()
pca.create(3, "linear")
params["file results"] = "PCA.csv"
dataset = dataset_model(params)
data = dataset.data.copy()
data = data.drop(columns=["entropy",
                          "label"])
data = data.to_numpy()
data = data @ data.T
data = (data-mean(data))/std(data)
pca.run(data)
results = pca.get_eigenvectors()
results["label"] = dataset.data["label"]
filename = join(params["path results"],
                params["file results"])
results.to_csv(filename,
               index=False)
