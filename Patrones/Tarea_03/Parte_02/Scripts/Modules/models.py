from sklearn.manifold import LocallyLinearEmbedding, Isomap
from sklearn.manifold import Isomap
from pandas import DataFrame
from numpy import array


class LLE_model_class:
    def __init__(self) -> None:
        self.obtain_index_plot_animals()

    def run(self, data: array) -> DataFrame:
        model = LocallyLinearEmbedding(n_components=2)
        results = model.fit_transform(data)
        self.results = {"x": results[:, 0],
                        "y": results[:, 1]}
        self.results = DataFrame(self.results)

    def obtain_index_plot_animals(self):
        self.animals_index = [19, 18, 2, 14, 15, 36, 38, 35, 31, 46]


class Isomap_model_class:
    def __init__(self, neighbors: int, components: int) -> None:
        self.components = components
        self.neighbors = neighbors
        self.obtain_index_plot_animals()

    def run(self, data: array) -> DataFrame:
        columns = ["Component {}".format(i+1)
                   for i in range(self.components)]
        model = Isomap(n_neighbors=self.neighbors,
                       n_components=self.components)
        model.fit_transform(data)
        results = model.transform(data)
        self.results = DataFrame(results,
                                 columns=columns)

    def obtain_index_plot_animals(self):
        self.animals_index = [30, 38, 12, 31, 26, 24, 35, 46, 19, 18]


class TSNE_model_class:
    def __init__(self) -> None:
        pass


class SOM_model_class:
    def __init__(self) -> None:
        pass
