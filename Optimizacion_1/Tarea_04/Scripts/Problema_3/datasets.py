from numpy import linspace


class datasets_class:
    """
    Conjunto de parametros para cada dataset
    """

    def __init__(self) -> None:

        self.datasets = {
            "lambda 1": {"path results": "../../Results/Problema_3/",
                         "n": 128,
                         "tau": 1e-6,
                         "lambda": 1,
                         "sigma": 1,
                         "c1": 0.1,
                         "c2": 0.9,
                         "problem name": "lambda",
                         "initial point": "predefined",
                         "test data": False,
                         },
            "lambda 10": {"path results": "../../Results/Problema_3/",
                          "n": 128,
                          "tau": 1e-6,
                          "lambda": 10,
                          "sigma": 10,
                          "c1": 0.1,
                          "c2": 0.9,
                          "problem name": "lambda",
                          "initial point": "predefined",
                          "test data": False,
                          },
            "lambda 1000": {"path results": "../../Results/Problema_3/",
                            "n": 128,
                            "tau": 1e-6,
                            "lambda": 1000,
                            "sigma": 1000,
                            "c1": 0.1,
                            "c2": 0.9,
                            "problem name": "lambda",
                            "initial point": "predefined",
                            "test data": False,
                            },
            "lambda 1 test": {"path results": "../../Results/Problema_3/",
                              "n": 128,
                              "tau": 1e-6,
                              "lambda": 1,
                              "sigma": 1,
                              "c1": 0.1,
                              "c2": 0.9,
                              "problem name": "lambda",
                              "initial point": "predefined",
                              "test data": True,
                              },
            "lambda 10 test": {"path results": "../../Results/Problema_3/",
                               "n": 128,
                               "tau": 1e-6,
                               "lambda": 10,
                               "sigma": 10,
                               "c1": 0.1,
                               "c2": 0.9,
                               "problem name": "lambda",
                               "initial point": "predefined",
                               "test data": True,
                               },
            "lambda 1000 test": {"path results": "../../Results/Problema_3/",
                                 "n": 128,
                                 "tau": 1e-6,
                                 "lambda": 1000,
                                 "sigma": 1000,
                                 "c1": 0.1,
                                 "c2": 0.9,
                                 "problem name": "lambda",
                                 "initial point": "predefined",
                                 "test data": True,
                                 }
        }

    def obtain_dataset(self, name: str) -> dict:
        return self.datasets[name]


class graphics_datasets_class:
    """
    Conjunto de parametros usados para realizar las graficas de cada dataset
    """

    def __init__(self) -> None:
        self.datasets = {"lambda 1 test": {"path data": "../../Results/Problema_3/",
                                           "path graphics": "../../Graphics/Problema_3/",
                                           "problem name": "lambda",
                                           "lambda": 1,
                                           "test data": True,
                                           "xlim": [-1, 1],
                                           "xticks": linspace(-1, 1, 11),
                                           "y lim": [80, 220], },
                         "lambda 10 test": {"path data": "../../Results/Problema_3/",
                                            "path graphics": "../../Graphics/Problema_3/",
                                            "problem name": "lambda",
                                            "test data": True,
                                            "lambda": 10,
                                            "xlim": [-1, 1],
                                            "xticks": linspace(-1, 1, 11),
                                            "y lim": [80, 220], },
                         "lambda 1000 test": {"path data": "../../Results/Problema_3/",
                                              "path graphics": "../../Graphics/Problema_3/",
                                              "problem name": "lambda",
                                              "test data": True,
                                              "lambda": 1000,
                                              "xlim": [-1, 1],
                                              "xticks": linspace(-1, 1, 11),
                                              "y lim": [80, 220]},
                         "lambda 1": {"path data": "../../Results/Problema_3/",
                                      "path graphics": "../../Graphics/Problema_3/",
                                      "problem name": "lambda",
                                      "lambda": 1,
                                      "test data": False,
                                      "xlim": [-1, 1],
                                      "xticks": linspace(-1, 1, 11),
                                      "y lim": [-3, 3], },
                         "lambda 10": {"path data": "../../Results/Problema_3/",
                                       "path graphics": "../../Graphics/Problema_3/",
                                       "problem name": "lambda",
                                       "test data": False,
                                       "lambda": 10,
                                       "xlim": [-1, 1],
                                       "xticks": linspace(-1, 1, 11),
                                       "y lim": [-25, 25], },
                         "lambda 1000": {"path data": "../../Results/Problema_3/",
                                         "path graphics": "../../Graphics/Problema_3/",
                                         "problem name": "lambda",
                                         "test data": False,
                                         "lambda": 1000,
                                         "xlim": [-1, 1],
                                         "xticks": linspace(-1, 1, 11),
                                         "y lim": [-2200, 2200]}, }

    def obtain_dataset(self, name: str) -> dict:
        return self.datasets[name]
