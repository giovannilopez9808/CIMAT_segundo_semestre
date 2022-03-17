from numpy import array, concatenate, savetxt
from .models import KMeans_model_class
from .functions import join_path
from numpy.random import normal
from pandas import DataFrame


class normal_model_class:
    def __init__(self) -> None:
        pass

    def create(self, mu: float, sigma: float, size: list) -> array:
        """
        Creacion del vector de puntos 
        """
        data = normal(mu, sigma, size=size)
        return data


class datasets_model:
    def __init__(self) -> None:
        self.data = self.create_points()
        self.validation = self.create_points()
        self.params = {"k values": range(2, 16),
                       }

    def create_points(self) -> array:
        """
        Creacion del dataset de puntos normales tridimensionales. 100 con la distribución 100 de N(4,1) y 100 de N(8,1)
        """
        normal_model = normal_model_class()
        mu4 = normal_model.create(4, 1, (100, 3))
        mu8 = normal_model.create(8, 1, (100, 3))
        data = concatenate((mu4, mu8))
        return data

    def run_kmeans(self):
        self.models = {}
        self.train_results = {}
        for k in self.params["k values"]:
            self.train_results[k] = {"labels": [],
                                     "scores": []}
            kmeans_model = KMeans_model_class()
            model = kmeans_model.run(k, self.data)
            self.models[k] = model
            self.train_results[k]["labels"] = model.labels_
            self.train_results[k]["scores"] = model.inertia_

    def run_kmeans_with_validation(self):
        self.validation_results = {}
        for k in self.params["k values"]:
            self.validation_results[k] = {"labels": [],
                                          "scores": []}
            model = self.models[k]
            self.validation_results[k]["scores"] = - \
                model.score(self.validation)
            self.validation_results[k]["labels"] = model.predict(
                self.validation)

    def save(self, params: dict):
        datasets = {"train": {"results": self.train_results,
                              "data": self.data,
                              "name": "train"},
                    "validation": {"results": self.validation_results,
                                   "data": self.validation,
                                   "name": "validation"}}
        for dataset_name in datasets:
            dataset = datasets[dataset_name]
            scores_filename = join_path(params["path results"],
                                        "{}_scores.csv".format(dataset["name"]))
            labels_filename = join_path(params["path results"],
                                        "{}_labels.csv".format(dataset["name"]))
            data_filename = join_path(params["path results"],
                                      "{}_data.csv".format(dataset["name"]))
            data = dataset["data"]
            scores = {}
            labels = {}
            for k in self.params["k values"]:
                scores[k] = dataset["results"][k]["scores"]
                labels[k] = dataset["results"][k]["labels"]
            scores = DataFrame(scores, index=["score"])
            scores = scores.T
            scores.index.name = "k"
            scores.to_csv(scores_filename)
            labels = DataFrame(labels)
            labels.index.name = "point"
            labels.to_csv(labels_filename)
            savetxt(data_filename,
                    data,
                    fmt="%.5f",
                    delimiter=",")
