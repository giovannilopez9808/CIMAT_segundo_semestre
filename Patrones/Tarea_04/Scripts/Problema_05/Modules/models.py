from sklearn.cluster import KMeans
from numpy import array


class KMeans_model_class:
    def __init__(self) -> None:
        pass

    def run(self, clusters: int,  data: array):
        model = KMeans(clusters, n_init=10)
        model.fit(data)
        return model
