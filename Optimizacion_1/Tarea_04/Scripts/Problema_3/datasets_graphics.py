from numpy import linspace


class graphics_datasets_class:
    def __init__(self) -> None:
        self.datasets = {"lambda 1": {"path data": "../../Results/Problema_3/",
                                      "path graphics": "../../Graphics/Problema_3/",
                                      "problem name": "lambda",
                                      "lambda": 1,
                                      "xlim": [-1, 1],
                                      "xticks": linspace(-1, 1, 11),
                                      "y lim": [80, 220], },
                         "lambda 10": {"path data": "../../Results/Problema_3/",
                                       "path graphics": "../../Graphics/Problema_3/",
                                       "problem name": "lambda",
                                       "lambda": 10,
                                       "xlim": [-1, 1],
                                       "xticks": linspace(-1, 1, 11),
                                       "y lim": [80, 220], },
                         "lambda 1000": {"path data": "../../Results/Problema_3/",
                                         "path graphics": "../../Graphics/Problema_3/",
                                         "problem name": "lambda",
                                         "lambda": 1000,
                                         "xlim": [-1, 1],
                                         "xticks": linspace(-1, 1, 11),
                                         "y lim": [80, 220]}}

    def obtain_dataset(self, name: str) -> dict:
        return self.datasets[name]
