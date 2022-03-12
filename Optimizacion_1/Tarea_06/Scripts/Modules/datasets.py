class datasets_class:
    """
    Conjunto de parametros para cada dataset
    """

    def __init__(self) -> None:
        self.datasets = {
            "log likehood bisection": {"path data": "../Data/",
                                       "file data": "mnist.pkl.gz",
                                       "path results": "../Results/",
                                       "tau": 1e-6,
                                       "c1": 1e-4,
                                       "c2": 0.9,
                                       "alpha": 0.001,
                                       "search name": "bisection",
                                       "problem name": "log likehood"}
        }

    def obtain_dataset(self, name: str) -> dict:
        return self.datasets[name]
