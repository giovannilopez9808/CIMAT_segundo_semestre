class datasets_class:
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