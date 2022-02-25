class datasets_class:
    """
    Conjunto de parametros para cada dataset
    """

    def __init__(self) -> None:
        self.datasets = {
            "rosembrock 2 predefined newton": {"path results": "../../Results/Problema_2/",
                                               "n": 2,
                                               "c1": 0.1,
                                               "c2": 0.9,
                                               "tau": 1e-6,
                                               "problem name": "rosembrock",
                                               "initial point": "predefined",
                                               "algorithm name": "newton"
                                               },
            "rosembrock 2 predefined gradient": {"path results": "../../Results/Problema_2/",
                                                 "n": 2,
                                                 "tau": 1e-6,
                                                 "alpha": 1e-3,
                                                 "problem name": "rosembrock",
                                                 "initial point": "predefined",
                                                 "algorithm name": "descent gradient"
                                                 },
            "rosembrock 2 random newton": {"path results": "../../Results/Problema_2/",
                                           "n": 2,
                                           "c1": 0.1,
                                           "c2": 0.9,
                                           "tau": 1e-7,
                                           "problem name": "rosembrock",
                                           "initial point": "random",
                                           "algorithm name": "newton"
                                           },
            "rosembrock 2 random gradient": {"path results": "../../Results/Problema_2/",
                                             "n": 2,
                                             "tau": 1e-6,
                                             "alpha": 1e-4,
                                             "problem name": "rosembrock",
                                             "initial point": "random",
                                             "algorithm name": "descent gradient"
                                             },
            "rosembrock 100 predefined gradient": {"path results": "../../Results/Problema_2/",
                                                   "n": 100,
                                                   "tau": 1e-6,
                                                   "alpha": 1e-3,
                                                   "problem name": "rosembrock",
                                                   "initial point": "predefined",
                                                   "algorithm name": "descent gradient"
                                                   },
            "rosembrock 100 predefined newton": {"path results": "../../Results/Problema_2/",
                                                 "n": 100,
                                                 "c1": 0.1,
                                                 "c2": 0.9,
                                                 "tau": 1e-12,
                                                 "problem name": "rosembrock",
                                                 "initial point": "predefined",
                                                 "algorithm name": "newton"
                                                 },
            "rosembrock 100 random gradient": {"path results": "../../Results/Problema_2/",
                                               "n": 100,
                                               "tau": 1e-8,
                                               "alpha": 1e-3,
                                               "problem name": "rosembrock",
                                               "initial point": "random",
                                               "algorithm name": "descent gradient"
                                               },
            "rosembrock 100 random newton": {"path results": "../../Results/Problema_2/",
                                             "n": 100,
                                             "c1": 0.1,
                                             "c2": 0.9,
                                             "tau": 1e-8,
                                             "problem name": "rosembrock",
                                             "initial point": "random",
                                             "algorithm name": "newton"
                                             },
            "wood 4 random newton": {"path results": "../../Results/Problema_2/",
                                     "n": 4,
                                     "c1": 0.1,
                                     "c2": 0.9,
                                     "tau": 1e-8,
                                     "problem name": "wood",
                                     "initial point": "random",
                                     "algorithm name": "newton"
                                     },
            "wood 4 random gradient": {"path results": "../../Results/Problema_2/",
                                       "n": 4,
                                       "tau": 1e-6,
                                       "alpha": 1e-4,
                                       "problem name": "wood",
                                       "initial point": "random",
                                       "algorithm name": "descent gradient"
                                       },
            "wood 4 predefined newton": {"path results": "../../Results/Problema_2/",
                                         "n": 4,
                                         "c1": 0.1,
                                         "c2": 0.9,
                                         "tau": 1e-6,
                                         "problem name": "wood",
                                         "initial point": "predefined",
                                         "algorithm name": "newton"
                                         },
            "wood 4 predefined gradient": {"path results": "../../Results/Problema_2/",
                                           "n": 4,
                                           "tau": 1e-6,
                                           "alpha": 1e-4,
                                           "problem name": "wood",
                                           "initial point": "predefined",
                                           "algorithm name": "descent gradient"
                                           }
        }

    def obtain_dataset(self, name: str) -> dict:
        return self.datasets[name]


class datasets_statistics_class:
    """
    Conjunto de parametros para cada dataset con la finalizad de hacer un analisis estadistico de los mismos
    """

    def __init__(self) -> None:
        self.datasets = {
            "wood 4 newton": {
                "path results": "../../Results/Problema_2/Random_results/",
                "path graphics": "../../Graphics/Problema_2/Random_results/",
                "filename": "wood_4_newton.csv",
                "filename graphics": "wood_4_newton.png",
            },
            "wood 4 gradient": {
                "path results": "../../Results/Problema_2/Random_results/",
                "path graphics": "../../Graphics/Problema_2/Random_results/",
                "filename": "wood_4_gradient.csv",
                "filename graphics": "wood_4_gradient.png"
            },
            "rosembrock 2 newton": {
                "path results": "../../Results/Problema_2/Random_results/",
                "path graphics": "../../Graphics/Problema_2/Random_results/",
                "filename": "rosembrock_2_newton.csv",
                "filename graphics": "rosembrock_2_newton.png"
            },
            "rosembrock 2 gradient": {
                "path graphics": "../../Graphics/Problema_2/Random_results/",
                "path results": "../../Results/Problema_2/Random_results/",
                "filename": "rosembrock_2_gradient.csv",
                "filename graphics": "rosembrock_2_gradient.png",
            },
            "rosembrock 100 newton": {
                "path graphics": "../../Graphics/Problema_2/Random_results/",
                "path results": "../../Results/Problema_2/Random_results/",
                "filename": "rosembrock_100_newton.csv",
                "filename graphics": "rosembrock_100_newton.png"
            },
            "rosembrock 100 gradient": {
                "path graphics": "../../Graphics/Problema_2/Random_results/",
                "path results": "../../Results/Problema_2/Random_results/",
                "filename": "rosembrock_100_gradient.csv",
                "filename graphics": "rosembrock_100_gradient.png",
            }}

    def obtain_dataset(self, name: str) -> dict:
        return self.datasets[name]


class graphics_datasets_class:
    """
    Conjunto de parametros usados para realizar las graficas de cada dataset
    """

    def __init__(self) -> None:
        self.datasets = {
            "wood 4 predefined": {
                "path data": "../../Results/Problema_2/",
                "path graphics": "../../Graphics/Problema_2/",
                "dataset": "wood",
                "initial point": "predefined",
                "n": 4,
                "xlim": [0, 50],
                "xticks": [x for x in range(0, 55, 5)],
                "y1 lim": [0, 4000],
                "y2 lim": [0, 3000],
            },
            "wood 4 random": {
                "path data": "../../Results/Problema_2/",
                "path graphics": "../../Graphics/Problema_2/",
                "dataset": "wood",
                "initial point": "random",
                "n": 4,
                "xlim": [0, 100],
                "xticks": [x for x in range(0, 105, 5)],
                "y1 lim": [0, 150],
                "y2 lim": [0, 500],
            },
            "rosembrock 2 predefined": {
                "path data": "../../Results/Problema_2/",
                "path graphics": "../../Graphics/Problema_2/",
                "dataset": "rosembrock",
                "initial point": "predefined",
                "n": 2,
                "xlim": [0, 50],
                "xticks": [x for x in range(0, 55, 5)],
                "y1 lim": [0, 30],
                "y2 lim": [0, 300],
            },
            "rosembrock 2 random": {
                "path data": "../../Results/Problema_2/",
                "path graphics": "../../Graphics/Problema_2/",
                "dataset": "rosembrock",
                "initial point": "random",
                "n": 2,
                "xlim": [0, 100],
                "xticks": [x for x in range(0, 105, 5)],
                "y1 lim": [0, 30],
                "y2 lim": [0, 300],
            },
            "rosembrock 100 predefined": {
                "path data": "../../Results/Problema_2/",
                "path graphics": "../../Graphics/Problema_2/",
                "dataset": "rosembrock",
                "initial point": "predefined",
                "n": 100,
                "xlim": [0, 100],
                "xticks": [x for x in range(0, 105, 5)],
                "y1 lim": [0, 30],
                "y2 lim": [0, 300],
            },
            "rosembrock 100 random": {
                "path data": "../../Results/Problema_2/",
                "path graphics": "../../Graphics/Problema_2/",
                "dataset": "rosembrock",
                "initial point": "random",
                "n": 100,
                "xlim": [0, 200],
                "xticks": [x for x in range(0, 220, 20)],
                "y1 lim": [0, 200],
                "y2 lim": [0, 300],
            }
        }

    def obtain_dataset(self, name: str) -> dict:
        return self.datasets[name]
