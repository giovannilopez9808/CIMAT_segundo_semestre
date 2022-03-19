from sklearn.cluster import KMeans
from numpy import array


class KMeans_model_class:
    def __init__(self) -> None:
        """
        Modelo de kmeans generalizado para su uso más eficiente
        """
        pass

    def run(self, clusters: int,  data: array):
        model = KMeans(clusters, n_init=10)
        model.fit(data)
        return model
