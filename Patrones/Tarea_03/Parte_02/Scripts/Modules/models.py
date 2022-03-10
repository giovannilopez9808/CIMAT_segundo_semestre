from sklearn.manifold import LocallyLinearEmbedding, Isomap, TSNE
from sklearn.manifold import Isomap
from pandas import DataFrame
from somlearn import SOM
from numpy import array


class LLE_model_class:
    def __init__(self) -> None:
        self.obtain_index_plot_animals()

    def run(self, data: array) -> DataFrame:
        model = LocallyLinearEmbedding(n_components=2)
        results = model.fit_transform(data)
        self.results = {"Component 1": results[:, 0],
                        "Component 2": results[:, 1]}
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
    def __init__(self, components: int) -> None:
        self.components = components
        self.obtain_index_plot_animals()

    def run(self, data: array) -> DataFrame:
        model = TSNE(n_components=self.components,
                     random_state=12345,
                     init="pca",
                     learning_rate="auto")
        results = model.fit_transform(data)
        self.results = {"Component 1": results[:, 0],
                        "Component 2": results[:, 1]}
        self.results = DataFrame(self.results)

    def obtain_index_plot_animals(self):
        self.animals_index = [30, 38, 12, 31,
                              26, 24, 35, 46,
                              19, 18, 2, 14, 15]


class SOM_model_class:
    def __init__(self) -> None:
        self.obtain_index_plot_animals()

    def run(self, data: array) -> DataFrame:
        model = SOM(n_columns=2,
                    n_rows=2,
                    random_state=1)
        self.results = model.fit_predict(data)

    def obtain_index_plot_animals(self):
        self.animals_index = [30, 38, 12, 31,
                              26, 24, 35, 46,
                              19, 18, 2, 14, 15]

    def create_classes_dataframe(self, names: array) -> DataFrame:
        self.classes = DataFrame(names,
                                 columns=["Names"])
        self.classes["Classes"] = self.results
        self.classes.index = self.classes["Names"]
        self.classes = self.classes.drop(columns="Names")
        self.classes = self.classes.sort_values("Classes")
