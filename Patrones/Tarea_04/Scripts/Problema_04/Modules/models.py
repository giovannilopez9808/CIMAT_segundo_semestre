from sklearn.cluster import KMeans
from numpy import array


class KMeans_model_class:
    def __init__(self) -> None:
        pass

    def run(self, clusters: int,  data: array, init: str = "k-means++"):
        n_init = 10
        if init in ["array"]:
            n_init = 1
            init = data[:clusters]
        model = KMeans(clusters,
                       n_init=n_init,
                       init=init)
        model.fit(data)
        return model
