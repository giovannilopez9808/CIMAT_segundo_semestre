from Modules.models import KMeans_model_class
from pandas import DataFrame, read_csv
from .functions import join_path
from numpy import array


def obtain_parms() -> dict:
    params = {"path data": "Data/",
              "path results": "Results/",
              "path graphics": "Graphics/",
              "k values": range(2, 7),
              "file data": "creditcard.csv"}
    return params


class image_class:
    def __init__(self, params: dict) -> None:
        self.k_values = params["k values"]
        self.params = params
        self.iterations = 5
        self.read()

    def read(self) -> array:
        filename = join_path(self.params["path data"],
                             self.params["file data"])
        data = read_csv(filename)
        self.classes = array(data["Class"].copy())
        data = data.drop(columns=["Class"])
        self.data = array(data)

    def run_kmeans(self) -> dict:
        kmean_model = KMeans_model_class()
        results = {}
        for k in self.k_values:
            print("Resolviendo {} cluster".format(k))
            results[k] = {"scores": [],
                          "predict": []}
            for i in range(self.iterations):
                model = kmean_model.run(k, self.data)
                results[k]["scores"] += [model.inertia_]
                results[k]["predict"] += [model.predict(self.data)]
        self.results = results

    def save(self) -> None:
        self.save_scores()
        self.save_labels()

    def save_scores(self) -> None:
        filename = join_path(self.params["path results"],
                             "scores.csv")
        data = {}
        for k in self.k_values:
            data[k] = self.results[k]["scores"]
        data = DataFrame(data)
        data.index.name = "Attempt"
        data.to_csv(filename)

    def save_labels(self) -> None:
        for k in self.k_values:
            filename = "cluster_{}.csv".format(k)
            filename = join_path(self.params["path results"],
                                 filename)
            data = array(self.results[k]["predict"])
            data = data.T
            data = DataFrame(data, columns=range(5))
            data["True"] = self.classes
            data.index.name = "id"
            data.to_csv(filename)
