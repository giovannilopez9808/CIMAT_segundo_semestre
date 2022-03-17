class datasets_class:
    """
    Conjunto de parametros para cada dataset
    """

    def __init__(self) -> None:
        self.datasets = {
            "log likelihood bisection": {"path data": "Data/",
                                         "file data": "mnist.pkl.gz",
                                         "path results": "Results/",
                                         "path graphics": "Graphics/",
                                         "tau": 1e-6,
                                         "c1": 1e-4,
                                         "c2": 0.9,
                                         "alpha": 0.001,
                                         "search name": "bisection",
                                         "problem name": "log likelihood"},
            # "log likelihood back tracking": {"path data": "../Data/",
            #                                  "file data": "mnist.pkl.gz",
            #                                  "path results": "../Results/",
            #                                  "tau": 1e-6,
            #                                  "rho": 0.9,
            #                                  "c1": 1e-4,
            #                                  "alpha": 0.001,
            #                                  "search name": "back tracking",
            #                                  "problem name": "log likelihood"},
        }

    def obtain_dataset(self, name: str) -> dict:
        return self.datasets[name]
